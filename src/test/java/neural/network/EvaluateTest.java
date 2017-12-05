package neural.network;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.ejml.data.FMatrixRMaj;
import org.junit.Before;
import org.junit.Test;

import neural.network.data.Image;
import neural.network.data.ImageReader;
import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;
import neural.network.exceptions.LayerTooSmallException;
import neural.network.exceptions.NetworkTooSmallException;

public class EvaluateTest {
	
	private Network network;
	private List<Image> testData;
	
	@Before
	public void setUp() throws NetworkTooSmallException, LayerTooSmallException, LayerDoesNotExistException, InvalidInputException, IOException {
		//Initialize a network
		network = new Network(Arrays.asList(new Integer(784), new Integer(15), new Integer(10)));
		//Initialize test Data
		testData = ImageReader.read("/mnist_test.csv");
	}
	
	@Test
	public void testEvaluate() throws InvalidInputException, LayerDoesNotExistException {
		System.out.println("Testing evaluate on a random network...");
		System.out.println("Result: " + Operations.evaluate(testData, network) + " / " + testData.size() + "\n");
	}
	
	@Test
	public void testMaxIndex(){
		System.out.println("Testing max index...");
		float data[][] = {{0.5f}, {0.25f}, {0.125f}, {0.0625f}, {1.0f}, {0.0f}, {-1.0f}};
		FMatrixRMaj vector = new FMatrixRMaj(data);
		assertEquals(4, Operations.maxIndex(vector));
	}
}
