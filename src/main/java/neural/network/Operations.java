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

	//Disallow instantiation of this class
	private Operations() {}
	
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
			final float eta, final boolean test) throws IOException, InvalidInputException, LayerDoesNotExistException {
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
				updateMiniBatch(network, mb, eta);
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
	 * @throws LayerDoesNotExistException 
	 * @throws InvalidInputException 
	 */
	private static void updateMiniBatch(final Network network, final List<Image> batch, final float eta) throws LayerDoesNotExistException, InvalidInputException {
		//Initialize gradient of biases to all zeros
		List<FMatrixRMaj> nablaB = new ArrayList<FMatrixRMaj>();
		for(FMatrixRMaj layer : network.getAllBiases()) {
			FMatrixRMaj entry = layer.copy();
			entry.zero();
			nablaB.add(entry);
		}
		
		//Initialize gradient of weights to all zeros
		List<FMatrixRMaj> nablaW = new ArrayList<FMatrixRMaj>();
		for(FMatrixRMaj layer : network.getAllWeights()) {
			FMatrixRMaj entry = layer.copy();
			entry.zero();
			nablaW.add(entry);
		}
		
		//Backpropogate
		for(Image img : batch) {
			//Change in nablaB and nablaW (i.e. Deltas)
			List<List<FMatrixRMaj>> result = backprop(network, img);
			//add change to nablaB and nablaW
			for(int i = 0; i < network.getNumberOfLayers()-1; i++) {
				FMatrixRMaj tempNablaB = nablaB.get(i).copy();
				CommonOps_FDRM.add(tempNablaB, result.get(0).get(i), tempNablaB);
				nablaB.set(i, tempNablaB);
				FMatrixRMaj tempNablaW = nablaW.get(i).copy();
				CommonOps_FDRM.add(tempNablaW, result.get(1).get(i), tempNablaW);
				nablaW.set(i, tempNablaW);
			}
		}
		
		//Update weights and biases
		for(int i = 0; i < network.getNumberOfLayers()-1; i++) {
			FMatrixRMaj newBiases = nablaB.get(i).copy();
			CommonOps_FDRM.scale(eta/batch.size(), newBiases);
			CommonOps_FDRM.subtract(network.getBiasesInLayer(i), newBiases, newBiases);
			network.setBiasInLayer(i, newBiases);
			FMatrixRMaj newWeights = nablaW.get(i).copy();
			CommonOps_FDRM.scale(eta/batch.size(), newWeights);
			CommonOps_FDRM.subtract(network.getWeightsInLayer(i), newWeights, newWeights);
			network.setWeightsInLayer(i, newWeights);
		}
	}
	
	/**
	 * Return a tuble ``(nabla_b, nabla_w)`` representing the gradient for the cost
	 * function C_x. ``nabla_b`` and ``nabla_w`` are EJML matrices, similar to
	 * ``self.biases`` and ``self.weights``.
	 * @throws LayerDoesNotExistException 
	 */
	private static List<List<FMatrixRMaj>> backprop(final Network network, final Image img) throws LayerDoesNotExistException {
		//Initialize gradient of biases to all zeros
		List<FMatrixRMaj> nablaB = new ArrayList<FMatrixRMaj>();
		for(FMatrixRMaj layer : network.getAllBiases()) {
			FMatrixRMaj entry = layer.copy();
			entry.zero();
			nablaB.add(entry);
		}
		
		//Initialize gradient of weights to all zeros
		List<FMatrixRMaj> nablaW = new ArrayList<FMatrixRMaj>();
		for(FMatrixRMaj layer : network.getAllWeights()) {
			FMatrixRMaj entry = layer.copy();
			entry.zero();
			nablaW.add(entry);
		}
		
		//Feedforward
		FMatrixRMaj activation = img.getVector();
		FMatrixRMaj expected = new FMatrixRMaj(network.getSizeOfLayers().get(network.getNumberOfLayers()-1),1);
		expected.zero();
		expected.set(img.getLabel(), 0, 1.0f);
		
		List<FMatrixRMaj> activations = new ArrayList<FMatrixRMaj>(); //List to store all the activations, layer by layer
		activations.add(activation);
		
		List<FMatrixRMaj> zs = new ArrayList<FMatrixRMaj>(); //List to store all the z vectors, layer by layer
		
		for(int i = 0; i < network.getNumberOfLayers()-1; i++) {
			FMatrixRMaj z = network.getBiasesInLayer(i).copy();
			CommonOps_FDRM.multAdd(network.getWeightsInLayer(i), activation, z);
			zs.add(z);
			activation = sigmoid(z);
			activations.add(activation);
		}
		
		//Backward pass
		FMatrixRMaj delta = nablaB.get(nablaB.size()-1).copy();
		CommonOps_FDRM.elementMult(cost_derivative(activations.get(activations.size()-1), expected), sigmoid_prime(zs.get(zs.size()-1)), delta);
		FMatrixRMaj deltaCopy = delta.copy(); //otherwise the value in the list changes later
		nablaB.set(nablaB.size()-1, deltaCopy);
		FMatrixRMaj w1 = network.getWeightsInLayer(network.getNumberOfLayers()-2).copy();
		FMatrixRMaj aT1 = activations.get(activations.size()-2).copy();
		CommonOps_FDRM.transpose(aT1);
		CommonOps_FDRM.mult(delta, aT1, w1);
		nablaW.set(nablaW.size()-1, w1);
		
		for(int i = 2; i <= nablaB.size(); i++) {
			FMatrixRMaj sp = sigmoid_prime(zs.get(zs.size()-i)).copy();
			FMatrixRMaj weightsTransposed = network.getWeightsInLayer(network.getNumberOfLayers()-i).copy();
			CommonOps_FDRM.transpose(weightsTransposed);
			FMatrixRMaj newDelta = new FMatrixRMaj(weightsTransposed.getNumRows(), delta.getNumCols());
			CommonOps_FDRM.mult(weightsTransposed, delta, newDelta);
			delta.reshape(sp.getNumRows(), newDelta.getNumCols());
			CommonOps_FDRM.elementMult(newDelta, sp, delta);
			nablaB.set(nablaB.size()-i, delta);
			
			FMatrixRMaj wi = network.getWeightsInLayer(network.getNumberOfLayers()-i-1).copy();
			FMatrixRMaj aTi = activations.get(activations.size()-i-1).copy();
			CommonOps_FDRM.transpose(aTi);
			CommonOps_FDRM.mult(delta, aTi, wi);
			nablaW.set(nablaW.size()-i, wi);
		}
		
		//Return
		List<List<FMatrixRMaj>> result = new ArrayList<List<FMatrixRMaj>>();
		result.add(nablaB);
		result.add(nablaW);
		return result;
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
	public static FMatrixRMaj cost_derivative(final FMatrixRMaj activation, final FMatrixRMaj expected) {
		FMatrixRMaj result = activation.copy();
		CommonOps_FDRM.subtract(activation, expected, result);
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
