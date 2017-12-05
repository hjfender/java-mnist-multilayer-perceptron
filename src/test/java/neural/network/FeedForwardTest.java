package neural.network;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.ejml.data.FMatrixRMaj;
import org.junit.Before;
import org.junit.Test;

import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;
import neural.network.exceptions.LayerTooSmallException;
import neural.network.exceptions.NetworkTooSmallException;

public class FeedForwardTest {

	private Network network;
	
	@Before
	public void setUp() throws NetworkTooSmallException, LayerTooSmallException, LayerDoesNotExistException, InvalidInputException {
		//Initialize network to predetermined values
		network = new Network(Arrays.asList(new Integer(3), new Integer(2), new Integer(3)));
		float[][] data1 = {{1.0f, 0.5f, 0.125f}, {0.25f, 0.03125f, 0.0625f}};
		network.setWeightsInLayer(0, new FMatrixRMaj(data1));
		float[][] data2 = {{1.0f, 0.5f}, {0.25f, 0.125f}, {0.03125f, 0.0625f}};
		network.setWeightsInLayer(1, new FMatrixRMaj(data2));
		float[][] data3 = {{1.0f}, {0.5f}};
		network.setBiasInLayer(0, new FMatrixRMaj(data3));
		float[][] data4 = {{-1.0f}, {-0.5f}, {-0.25f}};
		network.setBiasInLayer(1, new FMatrixRMaj(data4));
	}
	
	@Test
	public void testFeedForward() throws InvalidInputException, LayerDoesNotExistException {
		float data[][] = {{1.0f}, {1.0f}, {1.0f}};
		FMatrixRMaj input = new FMatrixRMaj(data);
		
		System.out.println("Testing Feed Forward...\n"+network+"\n");
		
		FMatrixRMaj output = Operations.feedForward(input, network);

		assertEquals(0.57005620, output.get(0,0), 0.000001);
		assertEquals(0.45525008, output.get(1,0), 0.000001);
	}
}