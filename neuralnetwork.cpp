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
#include <ctime>
#include <cstdlib>
using namespace std;


template <typename It>
void softmaxTemplate(It beg, It end)
{
	using VType = typename std::iterator_traits<It>::value_type;

	static_assert(std::is_floating_point<VType>::value,
		"Softmax function only applicable for floating types");

	auto max_ele{ *std::max_element(beg, end) };

	transform(
		beg,
		end,
		beg,
		[&](VType x) { return exp(x - max_ele); });

	VType exptot = std::accumulate(beg, end, 0.0);

	transform(
		beg,
		end,
		beg,
		[&](VType x) { auto ex = exp(x - max_ele); exptot += ex; return ex; });
}

struct Node {
	Node()=default;
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


template <typename T>
bool IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}


class MLPClassifier {
public:
	explicit MLPClassifier(int n_hidden_nodes, float lr, int lz, int ilz, int nc, int n_iter) : hidden_node_size(n_hidden_nodes) {learning_rate_sanity_check(lr) ? learning_rate = lr : learning_rate = 0.1f;
	hidden_layer_size = lz; input_layer_size = ilz; weight_size = ilz + 1; n_classes = nc; weight_size_final = n_hidden_nodes + 1; no_iterations = n_iter;
	};
	~MLPClassifier() {};
	using Vec = vector<float>;
	typedef vector<vector<Node>> layer;
	float learning_rate;
	int hidden_layer_size;
	int input_layer_size;
	int weight_size_final;
	int n_classes;
	int weight_size;
	int hidden_node_size;
	int no_iterations;
	bool learning_rate_sanity_check(float range) {
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
			Node mynode{this->weight_size_final};
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

	struct container {
		container(vector<Vec> t, vector<Vec> u, Vec v, Vec w, Vec x, Vec z) : W1(t), W2(u), A1(v), A2(w), Z1(x), Z2(z) {};
		vector<Vec> W1, W2;
		Vec A1, A2, Z1, Z2;
	};

	container forward_propagation(Vec training_data) {
		if (training_data.size() != this->input_layer_size) {
			training_data.emplace_back(0); // fit training data with bias term
		}
		
		Vec Z1;
		vector<Vec> W1;
		for (auto i=0; i < this->hidden_node_size; i++){
			W1.emplace_back(this->hidden_nodes[i].weights);
			Z1.emplace_back(static_cast<float>(inner_product(training_data.begin(), training_data.end(), std::begin(this->hidden_nodes[i].weights), 0.0)));
		}

		//for_each(firstweights.begin(), firstweights.end(), bind(&MLPClassifier::relu));
		auto A1 = this->relu_it(Z1);
		Vec Z2;
		vector<Vec> W2;
		for (auto i = 0; i < this->final_nodes.size(); i++) {
			W2.emplace_back(this->final_nodes[i].weights);
			Z2.emplace_back(static_cast<float>(inner_product(A1.begin(), A1.end(), std::begin(this->final_nodes[i].weights), 0.0)));
		}

		Vec A2 = this->softmaxoverflow(Z2);

		container cnt{ W1, W2, A1, A2, Z1, Z2 };
		return cnt;
	}

	Vec relu_it(Vec val) {
		auto size_vec = val.size();		
		Vec res;
		for (auto i = 0; i < size_vec; i++) {
			res.emplace_back(val[i] < 0.0f ? 0.0f : val[i]);
		}
		return res;
	}


	float relu(float val) {
		return val < 0.0f ? 0.0f : val;
	}

	float sigmoid(float x) {
		return exp(x) / (exp(x) + 1.0f);
	}


	auto loss_function_cross_entropy(Vec p, Vec q) { // p is ground truth, q is softmax/sigmoid predictions from forward prop
		Vec loss_vec;
		transform(p.begin(), p.end(), q.begin(), back_inserter(loss_vec), [&](float x, float y) {return x * log(y); });
		auto loss = accumulate(loss_vec.begin(), loss_vec.end(), 0.0);
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

	Vec softmaxoverflow(Vec weights) {
		Vec secondweights;
		Vec sum;
		float max = *std::max_element(weights.begin(), weights.end());

		for (auto i = 0; i < weights.size(); i++) {
			sum.emplace_back(exp(weights[i] - max));
		}

		auto norm2 = accumulate(sum.begin(), sum.end(), 0.0);

		for (auto i = 0; i < weights.size(); i++) {
			secondweights.emplace_back(exp(weights[i] - max) / norm2);
		}
		return secondweights;
	}

	Vec softmaxDerivative(Vec weights) {
		Vec derivativeWeights;
		Vec act = softmaxoverflow(weights);
		for (int i = 0; i < act.size(); i++) {
			derivativeWeights.emplace_back(act[i] * (1. - act[i]));
		}
		return derivativeWeights;
	}

	Vec relu_gradient(Vec dA, Vec Z) {
		Vec A = this->relu_it(Z);
		Vec B = this->setToZero(A);
		Vec dZ;
		transform(dA.begin(), dA.end(), A.begin(), std::back_inserter(dZ), std::multiplies<>{});
		return dZ;
	}

	pair<float, float> linear_backwards(Vec dZ, vector<Vec> W_curr, Vec weights_prev) {
		auto m = weights_prev.size();
		float dW = (1 / m) * inner_product(dZ.begin(), dZ.end(), activations_prev.begin(), 0.0);
		float dA_prev = inner_product(W_curr.begin(), W_curr.end(), dZ.begin(), 0.0);
		return make_pair(dA_prev, dW);
	}

	pair<float, float> linear_activation_backwards(Vec dA, vector<Vec> W_curr, Vec Z_curr, Vec A_prev, string activation_function="relu") {
		float da_prev, dW;
		if (activation_function == "relu") {
			auto dZ = this->relu_gradient(dA, Z_curr);
			tie(da_prev, dW) = linear_backwards(dZ, A_prev, W_curr);
		}
		else {
			auto dZ = this->softmaxDerivative(activation_weights);
			tie(da_prev, dW) = linear_backwards(dZ, A_prev, W_curr);
		}
		return make_pair(da_prev, dW);
	}

	Vec backwards_propagation(Vec actual, vector<float> training_data) {
		//tuple<Vec,Vec,vector<Vec>, Vec, Vec, vector<Vec>> res = this->forward_propagation(training_data);
		Vec grads;
		container ctr{ this->forward_propagation(training_data) };
		//Vec AL = get<0>(res);
		Vec preds = ctr.Z2;
		Vec result1;
		transform(std::begin(preds), std::begin(preds) + preds.size(), std::begin(actual),
			std::back_inserter(result1), [](float x, float y) {return x - y; });
		Vec result2;
		float myconstant{1};
		transform(preds.begin(), preds.end(), std::back_inserter(result2), [&myconstant](auto& c) {return (c)*(myconstant - c); });
		Vec dAL;
		transform(result1.begin(), result1.begin() + result1.size(), result2.begin(),
			std::back_inserter(dAL), std::divides<>{});
		float dA_prev_1, dW_prev_1;
		tie(dA_prev_1, dW_prev_1) = linear_activation_backwards(dAL, ctr.W2, ctr.A1, ctr.Z1, "softmax");


		//transform(v.begin(), v.end(), v.begin(), [k](int &c) { return c / k; });
		

		return AL;
	}


	Vec setToZero(Vec &a) {
		Vec b;
		for (auto i = a.begin(); i != a.end(); i++) {
			if (*i < 0.0f) {
				b.emplace_back(0);
			}
			else { b.emplace_back(*i); }
		}
		return b;
	}

};


int main() {

	MLPClassifier mynet(10, 0.3f, 10, 20, 3, 50);
	cout << "input layer size is " << mynet.input_layer_size << endl;
	cout << "my weights is " << mynet.weight_size << endl;
	cout << "final layer weights is " << mynet.weight_size_final << endl;
	cout << "learning rate is " << mynet.learning_rate << endl;
	vector<float> random_sample(mynet.input_layer_size);
	random_sample.emplace_back(0);
	generate(random_sample.begin(), random_sample.end(), [&]() {return static_cast <float> (rand() % 5);});
	for (auto &u : random_sample) {
		cout << "random " << u << endl;
	}

	mynet.initialise_parameters();
	vector<float> f = mynet.hidden_nodes[2].weights;
	for (auto &x : f) {
		cout << "weights " << x << endl;
	}
	auto firstprod = inner_product(random_sample.begin(), random_sample.end(), std::begin(mynet.hidden_nodes[0].weights), 0.0);
	cout << "first prod is " << firstprod << endl;
	auto prod = inner_product(std::begin(mynet.hidden_nodes[0].weights), std::end(mynet.hidden_nodes[0].weights), std::begin(mynet.final_nodes[0].weights), 0.0);
	cout << "inner product is " << prod << endl;
	auto allweights = mynet.forward_propagation(random_sample);


	using Vec = vector<float>;
	Vec a{ 1.2f, 1.4f, 1.6f };
	Vec b{ 3.3f, 0.5f, 2.4f };
	Vec result;
	Vec result2;
	transform(std::begin(a), std::begin(a) + a.size(), std::begin(b),
		std::back_inserter(result), std::divides<>{});
	for (auto &u : result) {
		cout << "division result " << u << endl;
	}
	transform(std::begin(a), std::begin(a) + a.size(), std::begin(b),
		std::back_inserter(result2), [](float x, float y) {return x - y; });
	for (auto &u : result2) {
		cout << "subtraction result " << u << endl;
	}

	std::vector<float> myarray;
	float myconstant{1};
	std::transform(a.begin(), a.end(), std::back_inserter(myarray), [&myconstant](auto& c) {return (c)*(myconstant-c); });
	for (auto &u : myarray) {
		cout << "multiply by 1 times itself " << u << endl;
	}
	Vec A{ -2, 0, 1, 3, 0 };
	Vec B{ 4,1,5,6,7 };
	Vec C = mynet.setToZero(A);
	Vec myres;
	transform(C.begin(), C.end(), A.begin(), std::back_inserter(myres), std::multiplies<>{});

	for (auto &b : myres) {
		cout << "vals " << b << endl;
	}

	auto m = mynet.hidden_nodes.size();
	cout << "m" << m << endl;
	auto dW = (1/m)*inner_product(A.begin(), A.end(), B.begin(), 0.0);
	cout << "dW " << dW << endl;


	//softmax(firstweights.begin(), firstweights.end());
	//for (auto &u : firstweights) {
	//	cout << "with softmax " << u << endl;
	//}




	/*vector<float> sense_check;
	float norm = accumulate(firstweights.begin(), firstweights.end(), 0.0);
	cout << "summed weights are " << norm << endl;


	//for_each(output.begin(), output.end(), [&](float x) { dvd(x); });

	
	//auto dotprod = mynet.linear_forward<Node>(mynet.hidden_nodes, mynet.final_nodes);
	/*for (auto &u : random_sample) {
		cout << "random " << u << endl;

	}

	mynet.populate_hidden_nodes();
	for (auto &u : mynet.initial_nodes) {
		cout << "hidden " << u << endl;
	}

	mynet.populate_final_nodes();
	for (auto &u : mynet.final_nodes) {
		cout << "final " << u << endl;
	}

	cout << "size is " << mynet.hidden_nodes.size() << endl;
	cout << mynet.sigmoid(1.5f) << endl;
	cout << mynet.relu(3.3f) << endl;
	vector<float> actual{ 0.2f, 0.5f, 0.3f };
	vector<float> preds{ 0.5f, 0.1f, 0.4f };
	float ans = mynet.loss_function_cross_entropy(actual, preds);
	cout << "cross entropy is " << ans << endl;
	vector<vector<Node>> layer;
	layer.push_back(mynet.hidden_nodes);*/

	return 0;
}