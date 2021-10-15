#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <cassert>
#include <ctime>
#include <cstdlib>
using namespace std;

using Vec = vector<float>;
template <typename It>
void softmax(It beg, It end)
{
	using VType = typename std::iterator_traits<It>::value_type;

	static_assert(std::is_floating_point<VType>::value,
		"Softmax function only applicable for floating types");

	auto max_ele{ *std::max_element(beg, end) };

	transform(beg, end, beg, [&](VType x) { return exp(x - max_ele); });

	VType exptot = std::accumulate(beg, end, 0.0);

	transform(beg,end, beg, [&](VType x) { auto ex = exp(x - max_ele); exptot += ex; return ex; });
}

struct Node {
	Node() = default;
	Node(int size) : weights(size) { sz = size; };
	Node(const Node& copynode) : weights(copynode.weights) { sz = copynode.sz; };
	int sz;
	vector<float> weights;
	friend ostream& operator <<(ostream& os, const Node nd) {
		os << "[";
		size_t last = nd.weights.size() - 1;
		for (size_t i = 0; i < nd.weights.size(); ++i) {
			os << nd.weights[i];
			if (i != last)
				os << ", ";
		}
		os << "]";
		return os;
	}
};

struct inputNode {
	inputNode(float val) : value(val) {};
	float value;
	friend ostream& operator <<(ostream& os, const inputNode nd) {
		return os << "[" << nd.value << "]";
	}
};

struct container {
	container() = default;
	container(Vec t, Vec u, Vec v, Vec w, Vec x, Vec z) : W1(t), W2(u), A1(v), A2(w), Z1(x), Z2(z) {};
	Vec W1, W2, A1, A2, Z1, Z2;
};

template <typename T>
bool constexpr IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}


class MLPClassifier {
public:
	explicit MLPClassifier(int n_hidden_nodes, float lr, int lz, int ilz, int nc, int n_iter, int n_hl) {
		hidden_node_size = n_hidden_nodes,
		learning_rate_sanity_check(lr) ? learning_rate = lr : learning_rate = 0.1f;
		hidden_layer_size = lz; input_layer_size = ilz; weight_size = ilz + 1; n_classes = nc; weight_size_final = n_hidden_nodes + 1; no_iterations = n_iter;
		no_hidden_layers = n_hl;
	};
	~MLPClassifier() {};
	typedef vector<vector<Node>> layer;
	float learning_rate;
	int hidden_layer_size;
	int input_layer_size;
	int weight_size_final;
	int n_classes;
	int weight_size;
	int hidden_node_size;
	int no_iterations;
	int no_hidden_layers;
	bool constexpr learning_rate_sanity_check(float range) {
		if (IsInBounds(range, 0.0f, 1.0f)) {
			return true;
		};
		return false;
	}
	vector<inputNode> initial_nodes;
	vector<Node> hidden_nodes;
	vector<Node> final_nodes;

	void enter_training_data(Vec data) {
		auto n_nodes = data.size();
		for (auto i = 0; i < n_nodes; i++) {
			inputNode curnode{ float(data.at(i)) };
			initial_nodes.push_back(curnode);
		}
	}

	void populate_hidden_nodes() {
		auto n_nodes = this->hidden_node_size;
		for (auto i = 0; i < n_nodes; i++) {
			Node mynode{ this->weight_size };
			generate_weights(mynode);
			hidden_nodes.push_back(mynode);
		}
	}

	void populate_final_nodes() {
		auto n_nodes = this->n_classes;
		for (auto i = 0; i < n_nodes; i++) {
			Node mynode{ this->weight_size_final };
			generate_weights(mynode);
			final_nodes.push_back(mynode);
		}
	}


	void generate_weights(Node& n) {
		if (n.weights.size() == 0) {
			throw exception("weights not initialised");
		}
		generate(n.weights.begin(), n.weights.end(), [&]() {return 0.1f * static_cast <float> (rand()) / static_cast <float> (RAND_MAX); });
	}

	void initialise_parameters() {
		initial_nodes.reserve(this->input_layer_size);
		populate_hidden_nodes();
		populate_final_nodes();
	}

	template<typename T>
	auto linear_forward(vector<T> prev, vector<T> next, int index) {
		return inner_product(std::begin(prev[index].weights), std::end(prev[index].weights), std::begin(next[index].weights), 0.0);
	}

	Vec matmul(const vector<Node>& W, const Vec& x, const Vec& b) {
		Vec z(W.size(), 0.0);
		for (unsigned int i = 0; i < W.size(); ++i) {
			for (unsigned int j = 0; j < W[0].weights.size(); ++j) {
				z[i] += W[i].weights[j] * x[j];
			}
			z[i] += b[i];
		}
		return z;
	}

	float double_dot_product(const vector<Node> &e, const Vec &p)
	{
		float result = 0;
		for (auto& v : e)
			result += std::inner_product(v.weights.begin(), v.weights.end(), p.begin(), 0.0);
		return result;
	}

	container forward_propagation(Vec& training_data) {

		Vec Z1, W1, Z2, W2;

		for (auto i = 0; i < this->hidden_nodes.size(); i++) {
			//Z1.emplace_back(double_dot_product(this->hidden_nodes, training_data));
			Z1.emplace_back(static_cast<float>(inner_product(training_data.begin(), training_data.end(), std::begin(this->hidden_nodes[i].weights), 0.0)));
			for (auto j = 0; j < this->hidden_nodes[0].sz; j++) {
				W1.emplace_back(this->hidden_nodes[i].weights[j]);
			}
		}

		auto A1 = this->relu_it(Z1);
		for (auto i = 0; i < this->final_nodes.size(); i++) {
			Z2.emplace_back(static_cast<float>(inner_product(A1.begin(), A1.end(), std::begin(this->final_nodes[i].weights), 0.0)));
			for (auto j = 0; j < this->final_nodes[0].sz; j++) {
				W2.emplace_back(this->final_nodes[i].weights[j]);
			}
		}
		Vec A2 = this->softmaxoverflow(Z2);
		container cnt{ W1, W2, A1, A2, Z1, Z2 };
		return cnt;
	}

	Vec relu_it(Vec& val) {
		auto size_vec = val.size();
		Vec res;
		for (auto i = 0; i < size_vec; i++) {
			cout << val[i] << endl;
			res.emplace_back(this->relu(val[i]));
			cout << this->relu(val[i]) << endl;
		}
		return res;
	}

	Vec convert_probs_to_class(Vec& probs) {
		Vec::iterator result = std::max_element(probs.begin(), probs.end());
		int argmaxVal = distance(probs.begin(), result);
		int selected_class = probs[argmaxVal];
		Vec one_hot_classes(probs.size());
		fill(one_hot_classes.begin(), one_hot_classes.end(), 0.0f);
		one_hot_classes[selected_class] = 1.0f;
		return one_hot_classes;
	}

	float relu(float val) {
		return val < 0.0f ? 0.0f : val;
	}

	float sigmoid(float x) { // will eventually be used for regression
		return exp(x) / (exp(x) + 1.0f);
	}


	float loss_function_cross_entropy(Vec& p, Vec& q) { // p is ground truth, q is softmax/sigmoid predictions from forward prop
		Vec loss_vec;
		transform(p.begin(), p.end(), q.begin(), back_inserter(loss_vec), [&](float x, float y) {return x * log(y) + float(1e-8); });
		float loss = accumulate(loss_vec.begin(), loss_vec.end(), 0.0f);
		return -loss;
	}

	void print_hidden_layer_weights(layer l) {
		for (unsigned int i = 0; i < l.size(); ++i)
		{
			for (unsigned int j = 0; j < l[i].size(); ++j)
			{
				cout << l[i][j];
			}
			cout << endl;
		}
	}

	Vec softmaxoverflow(Vec& weights) {
		Vec secondweights;
		Vec sum;
		float max = *std::max_element(weights.begin(), weights.end());

		for (auto i = 0; i < weights.size(); i++) {
			sum.emplace_back(exp(weights[i] - max));
		}

		float norm2 = accumulate(sum.begin(), sum.end(), 0.0);

		for (auto i = 0; i < weights.size(); i++) {
			secondweights.emplace_back(exp(weights[i] - max) / norm2);
		}
		return secondweights;
	}

	Vec softmaxDerivative(Vec& weights) {
		Vec derivativeWeights;
		Vec act = this->softmaxoverflow(weights);
		for (int i = 0; i < act.size(); i++) {
			derivativeWeights.emplace_back(act[i] * (1.f - act[i]));
		}
		return derivativeWeights;
	}

	Vec relu_gradient(Vec& dA, Vec& Z) {
		Vec A = this->relu_it(Z);
		Vec B = this->setToZero(A);
		Vec dZ;
		transform(dA.begin(), dA.end(), A.begin(), std::back_inserter(dZ), std::multiplies<>{});
		return dZ;
	}

	Vec setToZero(Vec &a) {
		Vec b;
		for (auto i = a.begin(); i != a.end(); i++) {
			if (*i < 0.0f) {
				b.emplace_back(0.0f);
			}
			else { b.emplace_back(*i); }
		}
		return b;
	}

	pair<float, float> linear_backwards(Vec& dZ, Vec& W_curr, Vec& weights_prev) {
		auto m = weights_prev.size();
		float dW = (1 / m) * inner_product(dZ.begin(), dZ.end(), weights_prev.begin(), 0.0);
		float(dA_prev) = inner_product(W_curr.begin(), W_curr.end(), dZ.begin(), 0.0);
		return make_pair(dA_prev, dW);
	}

	pair<float, float> linear_activation_backwards(Vec& dA, Vec& W_curr, Vec& Z_curr, Vec& A_prev, string activation_function = "relu") {
		float da_prev, dW;
		if (activation_function == "relu") {
			auto dZ = this->relu_gradient(dA, Z_curr);
			tie(da_prev, dW) = linear_backwards(dZ, W_curr, A_prev);
		}
		if (activation_function == "softmax") {
			auto dZ = this->softmaxDerivative(Z_curr);
			tie(da_prev, dW) = linear_backwards(dZ, W_curr, A_prev);
		}
		else { throw exception("only multi-class classification is supported at the moment."); }
		return make_pair(da_prev, dW);
	}


	Vec backwards_propagation(Vec& Y_hat, Vec& actual, Vec& training_data, container& ctr) {
		Vec grads, result1, result2, dA_prev;
		float myconstant{1};
		transform(std::begin(Y_hat), std::begin(Y_hat) + Y_hat.size(), std::begin(actual),
			std::back_inserter(result1), [](float x, float y) {return x - y; });
		transform(Y_hat.begin(), Y_hat.end(), std::back_inserter(result2), [&myconstant](auto& c) {return (c)*(myconstant - c); });
		transform(result1.begin(), result1.begin() + result1.size(), result2.begin(),
			std::back_inserter(dA_prev), std::divides<>{});
		for (auto i = 0; i < 2; i++) { // this implementation only has a single hidden layer for now
			auto dA_curr = dA_prev;
			float dA_prev, dW_prev;
			if (i==0){
				tie(dA_prev, dW_prev) = this->linear_activation_backwards(dA_curr, ctr.W2, ctr.Z2, ctr.A2, "softmax");}
			else {
				tie(dA_prev, dW_prev) = this->linear_activation_backwards(dA_curr, ctr.W1, ctr.Z1, ctr.A1, "relu");
			}
			grads.push_back(dW_prev);
		}
		return grads;
	}

	bool is_float_eq(float a, float b, float epsilon) {
		return ((a - b) < epsilon) && ((b - a) < epsilon);
	}

	bool get_accuracy_value(const Vec& Y_hat, const Vec& target) {
		assert(Y_hat.size() == target.size(), "predictions and targets must be same size");
		for (int i = 0; i < Y_hat.size(); i++) {
			if (int(target[i]) != int(0)) {
				if (int(target[i]) == int(Y_hat[i])) {
					return true;
				}
			}
		}
		return false;
	}

	void update_parameters(Vec & grads, const float learning_rate, vector<Node> & hidden_nodes) {
		int grads_size = grads.size();
		vector<Node> update_nodes;
		for (int i = 0; i < grads_size; i++) {
			for (int j = 0; j < hidden_nodes[0].weights.size(); j++) {
				//std::transform(hidden_nodes[i].weights.begin(), hidden_nodes[i].weights.end(), update_nodes[i].weights.begin(), [learning_rate, grads, i] { learning_rate - grads[i]; });
				this->hidden_nodes[i].weights[j] = this->hidden_nodes[i].weights[j] - this->learning_rate*grads[i];
			}
		}
	}
	pair<Vec, vector<bool>> train(Vec& training_data, Vec& target) {
		Vec cost_history;
		vector<bool> accuracy_history;
		for (auto i = 0; i < this->no_iterations; i++) {
			container ctr{this->forward_propagation(training_data)};
			Vec y_hat = this->convert_probs_to_class(ctr.A2);
			float cost = this->loss_function_cross_entropy(target, y_hat);
			bool accuracy = this->get_accuracy_value(y_hat, target);
			Vec grads = this->backwards_propagation(y_hat, training_data, target, ctr);
			this->update_parameters(grads, this->learning_rate, this->hidden_nodes);
			cost_history.push_back(cost);
			accuracy_history.push_back(accuracy);
		}
		return make_pair(cost_history, accuracy_history);
	}
};



int main() {

	MLPClassifier mynet(10, 0.3f, 10, 20, 3, 50, 1);
	cout << "input layer size is " << mynet.input_layer_size << endl;
	cout << "my weights is " << mynet.weight_size << endl;
	cout << "final layer weights is " << mynet.weight_size_final << endl;
	cout << "learning rate is " << mynet.learning_rate << endl;
	mynet.initialise_parameters();
	Vec classes = { 0.0f, 1.0f, 0.0f }; // one-hot encoded - using floats for now but will change to ints in future implementation
	Vec random_sample(mynet.input_layer_size);
	random_sample.emplace_back(0.0f);
	generate(random_sample.begin(), random_sample.end(), [&]() {return static_cast <float> (rand() % 5); });
	for (auto& u : random_sample) {
		cout << "sample " << u << endl;
	}
	container ctr{ mynet.forward_propagation(random_sample) };
	Vec y_hat = mynet.convert_probs_to_class(ctr.A2);
	for (auto& u: y_hat) {
		cout << "preds " << u << endl;
	}
	getchar();
	/*Vec grads = mynet.backwards_propagation(classes, y_hat, random_sample, ctr);
	for (auto &u : grads) {
		cout << "grads" << endl;
	}
	vector<float> f = mynet.hidden_nodes[2].weights;
	for (auto &x : f) {
		cout << "weights " << x << endl;
	}
	float norm = accumulate(firstweights.begin(), firstweights.end(), 0.0);
	cout << "summed weights are " << norm << endl;
	for_each(output.begin(), output.end(), [&](float x) { dvd(x); });*/

	return 0;
}