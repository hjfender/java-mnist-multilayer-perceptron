package neural.network;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.ejml.data.FMatrixRMaj;
import org.ejml.dense.row.CommonOps_FDRM;

import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;

/**
 * @author HEFE002
 * Encapsulates all the relevant operations on a multilayer perception.
 */
public class Operations {
	
	/**
	 * Compute elementwise sigmoid on a vector or matrix.
	 */
	public static FMatrixRMaj sigmoid(final FMatrixRMaj input) {
		final FMatrixRMaj output = input.copy();
		Sigmoid sigmoid = new Sigmoid();
		for(int i = 0; i < output.getNumRows(); i++) {
			for(int j = 0; j < output.getNumCols(); j++) {
				output.unsafe_set(i, j, (float) sigmoid.value(output.unsafe_get(i, j)));
			}
		}
		return output;
	}
	
	/**
	 * Compute the output of a neural network on a valid input.
	 * @throws InvalidInputException 
	 * @throws LayerDoesNotExistException 
	 */
	public static FMatrixRMaj feedForward(FMatrixRMaj input, final Network network) throws InvalidInputException, LayerDoesNotExistException {
		if(input.getNumRows() != network.getSizeOfLayers().get(0).intValue() || input.getNumCols() != 1) {
			throw new InvalidInputException(input);
		}
		FMatrixRMaj output = null;
		for(int i = 0; i < network.getNumberOfLayers()-2; i++) {
			output = new FMatrixRMaj(network.getSizeOfLayers().get(i+1),1);
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
	 * !!!(Translate this part to use the KNOWM dataset I found)!!!
	 * The "training_data" is a list of tuples "(x, y)" representing the training
	 * inputs and the desired outputs. The other non-optional parameters are self-explanatory.
	 * If "test_data" is provided then the network will be evaluated against the test data
	 * after each epoch, and partial progress printed out. This is useful for tracking
	 * progress, but slows things down substantially.
	 */
	public void stochasticGradientDescent() {
		//TODO: Implement
	}
	
	/**
	 * Update the network's weights and biases by applying gradient descent using
	 * backpropagation to a single mini batch.
	 * 
	 * !!!(Translate this part to use the KNOWM dataset I found)!!!
	 * The "mini batch" is a list of tuples "(x, y)", and "eta" is
	 * the learning rate.
	 */
	public void updateMiniBatch() {
		//TODO: Implement
	}
}
