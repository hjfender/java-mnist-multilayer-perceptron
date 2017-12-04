package neural.network;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.Arrays;
import java.util.List;

import org.ejml.data.FMatrixRMaj;
import org.junit.Test;

import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;
import neural.network.exceptions.LayerTooSmallException;
import neural.network.exceptions.NetworkTooSmallException;

public class NetworkTests {
	
	@Test
	public void testInitializationOfNetwork1() throws NetworkTooSmallException, LayerTooSmallException {
		List<Integer> sizes = Arrays.asList(new Integer(3), new Integer(2));
		Network network = new Network(sizes);
		System.out.println("Testing Initialization 1...\n"+network+"\n");
		assertEquals(network.getNumberOfLayers().intValue(), 2);
		assertEquals(network.getAllBiases().size(), 1);
		assertEquals(network.getAllWeights().size(), 1);
	}
	
	@Test
	public void testInitializationOfNetwork2() throws NetworkTooSmallException, LayerTooSmallException {
		List<Integer> sizes = Arrays.asList(new Integer(784), new Integer(15), new Integer(15), new Integer(10));
		Network network = new Network(sizes);
		System.out.println("Testing Initialization 2...\n"+network+"\n");
		assertEquals(network.getNumberOfLayers().intValue(), 4);
		assertEquals(network.getAllBiases().size(), 3);
		assertEquals(network.getAllWeights().size(), 3);
	}
	
	@Test
	public void testUpdateBiases() throws NetworkTooSmallException, LayerDoesNotExistException, LayerTooSmallException, InvalidInputException {
		List<Integer> sizes = Arrays.asList(new Integer(3), new Integer(2));
		Network network = new Network(sizes);
		System.out.println("Testing Biases...\n"+network+"\n");
		FMatrixRMaj previousBiases = network.getBiasesInLayer(0);
		FMatrixRMaj updatedBiases = previousBiases.copy();
		updatedBiases.set(0, 1000f);
		network.setBiasInLayer(0, updatedBiases);
		assertNotEquals(previousBiases.get(0), network.getBiasesInLayer(0).get(0));
		assertEquals(updatedBiases.get(0), network.getBiasesInLayer(0).get(0), 0.0);
	}
	
	@Test
	public void testWeights() throws NetworkTooSmallException, LayerDoesNotExistException, LayerTooSmallException, InvalidInputException {
		List<Integer> sizes = Arrays.asList(new Integer(3), new Integer(2));
		Network network = new Network(sizes);
		System.out.println("Testing Weights...\n"+network+"\n");
		FMatrixRMaj previousWeights = network.getWeightsInLayer(0);
		FMatrixRMaj updatedWeights = previousWeights.copy();
		updatedWeights.set(0, 0, 1000f);
		network.setWeightsInLayer(0, updatedWeights);
		assertNotEquals(previousWeights.get(0), network.getWeightsInLayer(0).get(0,0));
		assertEquals(updatedWeights.get(0), network.getWeightsInLayer(0).get(0,0), 0.0);
	}
	
	@Test(expected = LayerDoesNotExistException.class)
	public void testLayerDoesNotExist() throws NetworkTooSmallException, LayerDoesNotExistException, LayerTooSmallException {
		List<Integer> sizes = Arrays.asList(new Integer(3), new Integer(2));
		Network network = new Network(sizes);
		System.out.println("Testing LayerDoesNotExistException...\n");
		network.getBiasesInLayer(2).get(0);
	}
	
	@Test(expected = LayerTooSmallException.class)
	public void testLayerTooSmall() throws NetworkTooSmallException, LayerDoesNotExistException, LayerTooSmallException {
		System.out.println("Testing LayerTooSmallException...\n");
		List<Integer> sizes = Arrays.asList(new Integer(0), new Integer(-1));
		new Network(sizes);
	}
	
	@Test(expected = NetworkTooSmallException.class)
	public void testTooSmall() throws NetworkTooSmallException, LayerTooSmallException {
		List<Integer> sizes = Arrays.asList(new Integer(1));
		System.out.println("Testing NetworkTooSmallException...\n");
		new Network(sizes);
	}

}
