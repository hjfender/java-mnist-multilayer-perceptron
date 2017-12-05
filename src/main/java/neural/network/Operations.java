package neural.network;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.ejml.data.FMatrixRMaj;
import org.ejml.dense.row.CommonOps_FDRM;

import neural.network.data.Image;
import neural.network.data.ImageReader;
import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;

public class Operations {

	/**
	 * Compute elementwise sigmoid on a vector or matrix.
	 */
	public static FMatrixRMaj sigmoid(final FMatrixRMaj input) {
		final FMatrixRMaj output = input.copy();
		Sigmoid sigmoid = new Sigmoid();
		for (int i = 0; i < output.getNumRows(); i++) {
			for (int j = 0; j < output.getNumCols(); j++) {
				output.unsafe_set(i, j, (float) sigmoid.value(output.unsafe_get(i, j)));
			}
		}
		return output;
	}

	/**
	 * Compute the derivative of sigmoid elementwise on a vector or matrix.
	 */
	public static FMatrixRMaj sigmoid_prime(final FMatrixRMaj input) {
		final FMatrixRMaj temp = sigmoid(input).copy();
		for (int i = 0; i < temp.getNumRows(); i++) {
			for (int j = 0; j < temp.getNumCols(); j++) {
				temp.set(i, j, 1f - temp.get(i, j));
			}
		}
		CommonOps_FDRM.elementMult(sigmoid(input), temp, temp);
		return temp;
	}

	/**
	 * Compute the output of a neural network on a valid input.
	 * 
	 * @throws InvalidInputException
	 * @throws LayerDoesNotExistException
	 */
	public static FMatrixRMaj feedForward(FMatrixRMaj input, final Network network)
			throws InvalidInputException, LayerDoesNotExistException {
		if (input.getNumRows() != network.getSizeOfLayers().get(0).intValue() || input.getNumCols() != 1) {
			throw new InvalidInputException(input);
		}
		FMatrixRMaj output = null;
		for (int i = 0; i < network.getNumberOfLayers() - 1; i++) {
			output = new FMatrixRMaj(network.getSizeOfLayers().get(i + 1), 1);
			CommonOps_FDRM.mult(network.getWeightsInLayer(i), input, output);
			CommonOps_FDRM.add(output, network.getBiasesInLayer(i), output);
			output.set(sigmoid(output));
			input = output;
		}
		return output;
	}

	/**
	 * Train the neural network using mini-batch stochastic gradient descent.
	 * 
	 * The input is a list of image objects, which encapsulates the label and
	 * the pixel data of the training data. The other non-optional parameters
	 * are self-explanatory.
	 * 
	 * If "test_data" is provided then the network will be evaluated against the
	 * test data after each epoch, and partial progress printed out. This is
	 * useful for tracking progress, but slows things down substantially.
	 * 
	 * @throws IOException
	 * @throws LayerDoesNotExistException
	 * @throws InvalidInputException
	 * 
	 */
	public static void stochasticGradientDescent(final Network network, final int epochs, final int mini_batch_size,
			final int eta, final boolean test) throws IOException, InvalidInputException, LayerDoesNotExistException {
		// Import trainingData
		List<Image> trainingData = ImageReader.read("/mnist_train.csv");

		// Import testData if the user wants too
		List<Image> testData = null;
		if (test) {
			testData = ImageReader.read("/mnist_test.csv");
		}

		// Begin Gradient Descent
		for (int i = 0; i < epochs; i++) {
			// Randomize trainingData
			Collections.shuffle(trainingData);

			// Init mini Batches
			List<List<Image>> miniBatches = new LinkedList<List<Image>>();
			int tracker = 0;
			List<Image> miniBatch = new ArrayList<Image>();
			for (Image img : trainingData) {
				miniBatch.add(img);
				if (++tracker % mini_batch_size == 0) {
					miniBatches.add(miniBatch);
					miniBatch = new ArrayList<Image>();
				}
			}

			// Descend using mini batches
			for (List<Image> mb : miniBatches) {
				updateMiniBatch(mb, eta);
			}

			// Output status report
			if (test) {
				System.out.println("Epoch " + i + ": " + evaluate(testData, network) + " / " + testData.size());
			} else {
				System.out.println("Epoch " + i + " complete");
			}
		}
	}

	/**
	 * Update the network's weights and biases by applying gradient descent
	 * using backpropagation to a single mini batch.
	 * 
	 * The "mini batch" is a list of image objects and "eta" is the learning
	 * rate.
	 */
	private static void updateMiniBatch(final List<Image> batch, final int eta) {
		// TODO: Implement
	}
	
	/**
	 * Return a tuble ``(nabla_b, nabla_w)`` representing the gradient for the cost
	 * function C_x. ``nabla_b`` and ``nabla_w`` are EJML matrices, similar to
	 * ``self.biases`` and ``self.weights``.
	 */
	private static void backprop(final int[] data, final int label) {
		//TODO: Implement
	}

	/**
	 * Return the number of test inputs for which the neural network outputs the
	 * correct result. Note that the neural network's output is assumed to be
	 * the index of whichever neuron in the final layer has the highest
	 * activation.
	 * 
	 * @throws LayerDoesNotExistException
	 * @throws InvalidInputException
	 */
	public static int evaluate(List<Image> testData, final Network network)
			throws InvalidInputException, LayerDoesNotExistException {
		List<int[]> testResults = new LinkedList<int[]>();
		FMatrixRMaj testCase = new FMatrixRMaj(784, 1);
		//Compute the result of running the test data on the network
		//Store next to original label
		for (Image img : testData) {
			int[] pair = new int[2];
			for (int i = 0; i < testCase.getNumRows(); i++) {
				testCase.set(i, 0, img.getData()[i]);
			}
			pair[0] = maxIndex(feedForward(testCase, network));
			pair[1] = img.getLabel();
			testResults.add(pair);
		}
		//Count the number of successful test results
		int sum = 0;
		for(int[] tc : testResults) {
			if(tc[0]==tc[1]){
				sum++;
			}
		}
		return sum;
	}

	/**
	 * Return the vector of partial derivatives \partial C_x / \partial a for
	 * the output activations.
	 */
	public static FMatrixRMaj cost_derivative(final FMatrixRMaj output_activations, final FMatrixRMaj y) {
		FMatrixRMaj result = output_activations.copy();
		CommonOps_FDRM.subtract(output_activations, y, result);
		return result;
	}

	/**
	 * Returns the maximum index of a vector (vertical).
	 */
	public static int maxIndex(FMatrixRMaj vector) {
		assert(vector.getNumCols() == 1);
		float max = 0.0f;
		int index = -1;
		for(int i = 0; i < vector.getNumRows(); i++) {
			float val = vector.get(i,0);
			if(val > max){
				max = val;
				index = i;
			}
		}
		assert(index != -1);
		return index;
	}
}
