package neural.network;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.ejml.data.FMatrixRMaj;

import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;
import neural.network.exceptions.LayerTooSmallException;
import neural.network.exceptions.NetworkTooSmallException;

/**
 * @author HEFE002
 * Encapsulates the properties and initialization of a multilayer perceptron.
 */
public class Network {

	private static final RandomDataGenerator GAUSSIAN_GENERATOR = new RandomDataGenerator();
	private final Integer numberOfLayers;
	private final List<Integer> sizeOfLayers;
	private final List<FMatrixRMaj> biases;
	private final List<FMatrixRMaj> weights;
	
	/**
	 * Initialize the neural network with randomly chosen weights and biases from a given
	 * list of layer sizes.
	 * 
	 * Weights are chosen by a Gaussian distribution.
	 * @throws LayerTooSmallException 
	 */
	public Network(final List<Integer> sizeOfLayers) throws NetworkTooSmallException, LayerTooSmallException {
		this.numberOfLayers = sizeOfLayers.size();
		if(numberOfLayers<=1){
			throw new NetworkTooSmallException();
		}
		this.sizeOfLayers = sizeOfLayers;
		this.biases = generateBiases();
		this.weights = getWeights();
	}

	private List<FMatrixRMaj> generateBiases() throws LayerTooSmallException {
		final List<FMatrixRMaj> biases = new ArrayList<FMatrixRMaj>();
		for(int i = 1; i < numberOfLayers; i++) {
			if(sizeOfLayers.get(i)<=0){
				throw new LayerTooSmallException();
			}
			final FMatrixRMaj layerBias = new FMatrixRMaj(sizeOfLayers.get(i), 1);
			for(int j = 0; j < layerBias.getNumRows(); j++){
				layerBias.set(j, 0, (float) GAUSSIAN_GENERATOR.nextGaussian(0,1));
			}
			biases.add(layerBias);
		}
		return biases;
	}
	
	private List<FMatrixRMaj> getWeights() throws LayerTooSmallException {
		final List<FMatrixRMaj> weights = new ArrayList<FMatrixRMaj>();
		for(int i = 0; i < numberOfLayers-1; i++) {
			if(sizeOfLayers.get(i)<=0){
				throw new LayerTooSmallException();
			}
			final FMatrixRMaj layerWeights = new FMatrixRMaj(sizeOfLayers.get(i+1), sizeOfLayers.get(i));
			for(int j = 0; j < layerWeights.getNumRows(); j++){
				for(int k = 0; k < layerWeights.getNumCols(); k++) {
					layerWeights.set(j, k, (float) GAUSSIAN_GENERATOR.nextGaussian(0,1));
				}
			}
			weights.add(layerWeights);
		}
		return weights;
	}

	public Integer getNumberOfLayers() {
		return numberOfLayers;
	}

	public List<Integer> getSizeOfLayers() {
		return sizeOfLayers;
	}

	public List<FMatrixRMaj> getAllBiases() {
		return biases;
	}
	
	public FMatrixRMaj getBiasesInLayer(int layer) throws LayerDoesNotExistException {
		if(layer < 0 || layer >= numberOfLayers.intValue()) {
			throw new LayerDoesNotExistException(layer);
		}
		return biases.get(layer);
	}

	public void setBiasInLayer(int layer, FMatrixRMaj biases) throws LayerDoesNotExistException, InvalidInputException {
		if(layer < 0 || layer >= numberOfLayers.intValue()) {
			throw new LayerDoesNotExistException(layer);
		}
		if(biases.getNumRows() != this.biases.get(layer).getNumRows() || biases.getNumCols() != 1){
			throw new InvalidInputException(biases);
		}
		this.biases.set(layer, biases);
	}
	
	public List<FMatrixRMaj> getAllWeights() {
		return weights;
	}
	
	public FMatrixRMaj getWeightsInLayer(int layer) throws LayerDoesNotExistException {
		if(layer < 0 || layer >= numberOfLayers.intValue()) {
			throw new LayerDoesNotExistException(layer);
		}
		return weights.get(layer);
	}

	public void setWeightsInLayer(int layer, FMatrixRMaj weights) throws LayerDoesNotExistException, InvalidInputException {
		if(layer < 0 || layer >= numberOfLayers.intValue()) {
			throw new LayerDoesNotExistException(layer);
		}
		if(weights.getNumRows() != this.weights.get(layer).getNumRows() || weights.getNumCols() != this.weights.get(layer).getNumCols()){
			throw new InvalidInputException(weights);
		}
		this.weights.set(layer, weights);
	}
	
	@Override
	public String toString() {
		try {
			return "Network: " + super.toString() +
				   ", Number of Layers: " + this.numberOfLayers + 
				   ", Size of Bias List: " + this.biases.size() +
				   ", Size of Weight List: " + this.biases.size() +
				   ", Random Layer First Bias: " +
				   this.getBiasesInLayer(ThreadLocalRandom.current().nextInt(0, this.numberOfLayers-1)).get(0) +
				   ", Random Layer Top Left Weight in Matrix: " +
				   this.getWeightsInLayer(ThreadLocalRandom.current().nextInt(0, this.numberOfLayers-1)).get(0, 0);
		} catch (LayerDoesNotExistException e) {
			e.printStackTrace();
		}
		return "ERROR";
	}
	
}
