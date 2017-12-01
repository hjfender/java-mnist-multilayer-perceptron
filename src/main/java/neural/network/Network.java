package neural.network;

import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.ejml.data.FMatrixRMaj;

public class Network {

	private static final RandomDataGenerator GAUSSIAN_GENERATOR = new RandomDataGenerator();
	private final Integer numberOfLayers;
	private final List<Integer> sizeOfLayers;
	private final List<FMatrixRMaj> biases;
	private final List<FMatrixRMaj> weights;
	
	public Network(final List<Integer> sizeOfLayers) {
		this.numberOfLayers = sizeOfLayers.size();
		this.sizeOfLayers = sizeOfLayers;
		this.biases = generateBiases();
		this.weights = getWeights();
	}

	private List<FMatrixRMaj> generateBiases() {
		final List<FMatrixRMaj> biases = new ArrayList<FMatrixRMaj>();
		for(int i = 1; i < numberOfLayers; i++) {
			final FMatrixRMaj layerBias = new FMatrixRMaj(sizeOfLayers.get(i),1);
			for(int j = 0; j < layerBias.getNumRows(); j++){
				layerBias.set(j, 0, (float) GAUSSIAN_GENERATOR.nextGaussian(0,1));
			}
			biases.set(i-1, layerBias);
		}
		return biases;
	}
	
	private List<FMatrixRMaj> getWeights() {
		final List<FMatrixRMaj> weights = new ArrayList<FMatrixRMaj>();
		for(int i = 0; i < numberOfLayers-1; i++) {
			final FMatrixRMaj layerBias = new FMatrixRMaj(sizeOfLayers.get(i),sizeOfLayers.get(i+1));
			for(int j = 0; j < layerBias.getNumRows(); j++){
				for(int k = 0; k < layerBias.getNumCols(); k++) {
					layerBias.set(j, k, (float) GAUSSIAN_GENERATOR.nextGaussian(0,1));
				}
			}
			biases.set(i-1, layerBias);
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
	
	public FMatrixRMaj getBiasesInLayer(int layer) {
		return biases.get(layer);
	}

	public void setBiasesInLayer(int layer, FMatrixRMaj biases) {
		this.biases.set(layer, biases);
	}
	
	public List<FMatrixRMaj> getAllWeights() {
		return weights;
	}
	
	public FMatrixRMaj getWeightsInLayer(int layer) {
		return weights.get(layer);
	}

	public void setWeightsInLayer(int layer, FMatrixRMaj weights) {
		this.weights.set(layer, weights);
	}
	
	@Override
	public String toString() {
		return "Network: " + super.toString() +
			   ", Number of Layers: " + this.numberOfLayers;
	}
	
}
