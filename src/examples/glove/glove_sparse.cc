/*
 * An example program that takes a GloVe
 * (http://nlp.stanford.edu/projects/glove/) dataset and builds a cross-polytope
 * LSH table with the following goal in mind: for a random subset of NUM_QUERIES
 * points, we would like to find a nearest neighbor (w.r.t. cosine similarity)
 * with probability at least 0.9.
 *
 * There is a function get_default_parameters, which you can use to set the
 * parameters automatically (in the code, we show how it could have been used).
 * However, we recommend to set parameters manually to maximize the performance.
 *
 * You need to specify:
 *   - NUM_HASH_TABLES, which affects the memory usage: the larger it is, the
 *     better (unless it's too large). Despite that, it's usually a good idea
 *     to start with say 10 tables, and then increase it gradually, while
 *     observing the effect it makes.
 *   - NUM_HASH_BITS, that controls the number of buckets per table,
 *     usually it should be around the binary logarithm of the number of data
 *     points
 *   - NUM_ROTATIONS, which controls the number of pseudo-random rotations for
 *     the cross-polytope LSH, set it to 1 for the dense data, and 2 for the
 *     sparse data (for GloVe we set it to 1)
 *
 * The code sets the number of probes automatically. Also, it recenters the
 * dataset for improved partitioning. Since after recentering vectors are not
 * unit anymore we should use the Euclidean distance in the data structure.
 */

#include <falconn/lsh_nn_table.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cstdio>
#include <cmath>
#include <fstream>

#include <sys/stat.h>
#include <unistd.h>
#include <string>

using std::fmax;
using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::make_pair;
using std::max;
using std::mt19937_64;
using std::pair;
using std::runtime_error;
using std::string;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::SparseVector ;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;
typedef SparseVector<float,int> Point;
using falconn::core::CosineDistanceSparse;
CosineDistanceSparse<float> distance_function;
const string FILE_NAME = "dataset/glove.840B.300d_sparse.dat";
const int NUM_QUERIES = 1000;
const int SEED = 4057218;
const int NUM_HASH_TABLES = 50;
const int NUM_HASH_BITS = 18;
const int NUM_ROTATIONS = 2;

/*
 * An auxiliary function that reads a point from a binary file that is produced
 * by a script 'prepare-dataset.sh'
 */

void persist(vector<int> v, string file_name) {
  std::ofstream output_file(file_name);
  for (const auto &e : v) output_file << e << " ";

}

inline bool exists_file(const std::string& name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

bool read_point(FILE *file, Point *point) {
  int d;
  if (fread(&d, sizeof(int), 1, file) != 1) {
    return false;
  }
  //std::vector<std::pair<float, int>> *  point_created = new std::vector<std::pair<float, int>>();
  int * int_buf = new int[d];
  if (fread(int_buf, sizeof(int), d, file) != (size_t)d) {
    throw runtime_error("can't read a point");
  }
  float * float_buf = new float[d];
  if (fread(float_buf, sizeof(float), d, file) != (size_t)d) {
      throw runtime_error("can't read a point");
  }
  point->resize(d);
  for (int i = 0; i < d; ++i) {
    (*point)[i] = std::make_pair(int_buf[i],float_buf[i]);
  }
  delete[] int_buf;
  delete[] float_buf;
  return true;
}

/*
 * An auxiliary function that reads a dataset from a binary file that is
 * produced by a script 'prepare-dataset.sh'
 */
void read_dataset(string file_name, vector<Point> *dataset) {
  FILE *file = fopen(file_name.c_str(), "rb");
  if (!file) {
    throw runtime_error("can't open the file with the dataset");
  }
  Point p;
  dataset->clear();
  while (read_point(file, &p)) {
    dataset->push_back(p);
  }
  if (fclose(file)) {
    throw runtime_error("fclose() error");
  }
}

/*
 * Normalizes the dataset.
 */
void normalize(vector<Point> *dataset) {
  for (size_t j=0; j < (*dataset).size();j++) {
    float denominator = 0.0;
    for(size_t i=0; i< (*dataset)[j].size();i++) {
      denominator += ((*dataset)[j][i].second) * ((*dataset)[j][i].second);
    }
    denominator = sqrt(denominator);
    if(denominator > 0) {
      for(size_t i=0; i< (*dataset)[j].size();i++) {
        (*dataset)[j][i].second /= denominator;
      }
    }
  }
}

/*
 * Chooses a random subset of the dataset to be the queries. The queries are
 * taken out of the dataset.
 */
void gen_queries(vector<Point> *dataset, vector<Point> *queries) {
  mt19937_64 gen(SEED);
  queries->clear();
  for (int i = 0; i < NUM_QUERIES; ++i) {
    uniform_int_distribution<> u(0, dataset->size() - 1);
    int ind = u(gen);
    queries->push_back((*dataset)[ind]);
    (*dataset)[ind] = dataset->back();
    dataset->pop_back();
  }
}

/*
 * Generates answers for the queries using the (optimized) linear scan.
 */
void gen_answers(const vector<Point> &dataset, const vector<Point> &queries,
                 vector<int> *answers) {
  answers->resize(queries.size());
  int outer_counter = 0;
  for (const auto &query : queries) {
    float best = -10.0;
    int inner_counter = 0;
    for (const Point &datapoint : dataset) {
      //float score = query.dot(datapoint);
      float score = -distance_function(query, datapoint);
      if (score > best) {
        (*answers)[outer_counter] = inner_counter;
        best = score;
      }
      ++inner_counter;
    }
    ++outer_counter;
  }
}

/*
 * Computes the probability of success using a given number of probes.
 */
double evaluate_num_probes(LSHNearestNeighborTable<Point> *table,
                           const vector<Point> &queries,
                           const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  int outer_counter = 0;
  int num_matches = 0;
  vector<int32_t> candidates;
  for (const auto &query : queries) {
    query_object->get_candidates_with_duplicates(query, &candidates);
    for (auto x : candidates) {
      if (x == answers[outer_counter]) {
        ++num_matches;
        break;
      }
    }
    ++outer_counter;
  }
  return (num_matches + 0.0) / (queries.size() + 0.0);
}

/*
 * Queries the data structure using a given number of probes.
 * It is much slower than 'evaluate_num_probes' and should be used to
 * measure the time.
 */
pair<double, QueryStatistics> evaluate_query_time(
    LSHNearestNeighborTable<Point> *table, const vector<Point> &queries,
    const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  query_object->reset_query_statistics();
  int outer_counter = 0;
  int num_matches = 0;
  for (const auto &query : queries) {
    if (query_object->find_nearest_neighbor(query) == answers[outer_counter]) {
      ++num_matches;
    }
    ++outer_counter;
  }
  return make_pair((num_matches + 0.0) / (queries.size() + 0.0),
                   query_object->get_query_statistics());
}

/*
 * Finds the smallest number of probes that gives the probability of success
 * at least 0.9 using binary search.
 */
int find_num_probes(LSHNearestNeighborTable<Point> *table,
                    const vector<Point> &queries, const vector<int> &answers,
                    int start_num_probes) {
  int num_probes = start_num_probes;
  for (;;) {
    cout << "trying " << num_probes << " probes" << endl;
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    if (precision >= 0.9) {
      break;
    }
    num_probes *= 2;
  }

  int r = num_probes;
  int l = r / 2;

  while (r - l > 1) {
    int num_probes = (l + r) / 2;
    cout << "trying " << num_probes << " probes" << endl;
    double precision = evaluate_num_probes(table, queries, answers, num_probes);
    if (precision >= 0.9) {
      r = num_probes;
    } else {
      l = num_probes;
    }
  }

  return r;
}

int main() {
  try {
    vector<Point> dataset, queries;
    vector<int> answers;

    // read the dataset
    cout << "reading points" << endl;
    read_dataset(FILE_NAME, &dataset);
    cout << dataset.size() << " points read" << endl;

    // normalize the data points
    cout << "normalizing points" << endl;
    normalize(&dataset);
    cout << "done" << endl;

    // find the center of mass
    Point * center = new Point();
    center->resize(300);
    for(int i=0;i<300;i++) {
      (*center)[i] = std::make_pair(i,0);
    }
    for (size_t i = 0; i < dataset.size(); ++i) {
      for(size_t j=0; j < dataset[i].size();j++) {
        int idx = dataset[i][j].first;
        float num = dataset[i][j].second;
        (*center)[idx].second = (*center)[idx].second + num;
      }
    }
    for(int i=0;i<300;i++) {
      (*center)[i].second /= dataset.size();
    }

    // selecting NUM_QUERIES data points as queries
    cout << "selecting " << NUM_QUERIES << " queries" << endl;
    gen_queries(&dataset, &queries);
    cout << "done" << endl;

    // running the linear scan
    cout << "running linear scan (to generate nearest neighbors)" << endl;
    auto t1 = high_resolution_clock::now();
    if(exists_file("answers.txt")) {
      std::ifstream input_file("answers.txt");
      int tmp_int;
      while(input_file >> tmp_int) {
        answers.push_back(tmp_int);
      }

    } else {
      gen_answers(dataset, queries, &answers);
      persist(answers,"answers.txt");
    }
    auto t2 = high_resolution_clock::now();
    double elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "done" << endl;
    cout << elapsed_time / queries.size() << " s per query" << endl;

    // re-centering the data to make it more isotropic
    cout << "re-centering" << endl;
    for (size_t i = 0; i < dataset.size(); ++i) {
      for(size_t j=0; j < dataset[i].size();j++) {
        int idx = dataset[i][j].first;
          dataset[i][j].second -= (*center)[idx].second;
      }
    }

    for (size_t i = 0; i < queries.size(); ++i) {
      for(size_t j=0; j < queries[i].size();j++) {
        int idx = queries[i][j].first;
        queries[i][j].second -= (*center)[idx].second;
      }
    }

//    for (auto &datapoint : dataset) {
//      datapoint -= center;
//    }
//    for (auto &query : queries) {
//      query -= center;
//    }



    cout << "done" << endl;

    // setting parameters and constructing the table
//    LSHConstructionParameters params;
//    params.dimension = dataset[0].size();
//    params.lsh_family = LSHFamily::CrossPolytope;
//    params.l = NUM_HASH_TABLES;
//    params.distance_function = DistanceFunction::EuclideanSquared;
//    compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
//    params.num_rotations = NUM_ROTATIONS;
//    // we want to use all the available threads to set up
//    params.num_setup_threads = 0;
//    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
   //   For an easy way out, you could have used the following.

//    LSHConstructionParameters params
//      = get_default_parameters<Point>(dataset.size(),
//                                 300,
//                                 DistanceFunction::NegativeInnerProduct,
//                                 true);
//    cout << "building the index based on the cross-polytope LSH" << endl;
//    params.feature_hashing_dimension=256;

    LSHConstructionParameters params;
    params.dimension = dataset[0].size();
    params.lsh_family = LSHFamily::Hyperplane;
    params.l = NUM_HASH_TABLES;
    params.distance_function = DistanceFunction::EuclideanSquared;
    compute_number_of_hash_functions<Point>(NUM_HASH_BITS, &params);
    //params.num_rotations = NUM_ROTATIONS;
    // we want to use all the available threads to set up
    params.num_setup_threads = 0;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    params.feature_hashing_dimension = 256;


    t1 = high_resolution_clock::now();
    auto table = construct_table<Point>(dataset, params);
    t2 = high_resolution_clock::now();
    elapsed_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "done" << endl;
    cout << "construction time: " << elapsed_time << endl;

    // finding the number of probes via the binary search
    cout << "finding the appropriate number of probes" << endl;
    int num_probes = find_num_probes(&*table, queries, answers, params.l);
    cout << "done" << endl;
    cout << num_probes << " probes" << endl;
    //int num_probes = 500;
    // executing the queries using the found number of probes to gather
    // statistics
    auto tmp = evaluate_query_time(&*table, queries, answers, num_probes);
    auto score = tmp.first;
    auto statistics = tmp.second;
    cout << "average total query time: " << statistics.average_total_query_time
         << endl;
    cout << "average lsh time: " << statistics.average_lsh_time << endl;
    cout << "average hash table time: " << statistics.average_hash_table_time
         << endl;
    cout << "average distance time: " << statistics.average_distance_time
         << endl;
    cout << "average number of candidates: "
         << statistics.average_num_candidates << endl;
    cout << "average number of unique candidates: "
         << statistics.average_num_unique_candidates << endl;
    cout << "score: " << score << endl;
  } catch (runtime_error &e) {
    cerr << "Runtime error: " << e.what() << endl;
    return 1;
  } catch (exception &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  } catch (...) {
    cerr << "ERROR" << endl;
    return 1;
  }
  return 0;
}
