package neural.network;

import static org.junit.Assert.assertEquals;

import org.ejml.data.FMatrixRMaj;
import org.junit.Test;

public class SigmoidTest {
	
	@Test
	public void testSigmoid() {
		float data[][] = {{1.0f}, {0.5f}, {0.25f}, {0.125f}, {0.0625f}, {0.0f}, {-1.0f}};
		FMatrixRMaj input = new FMatrixRMaj(data);
		
		System.out.println("Testing Sigmoid...\n");
		
		FMatrixRMaj output = Operations.sigmoid(input);
		
		assertEquals(0.7310585, output.get(0,0), 0.000001);
		assertEquals(0.6224593, output.get(1,0), 0.000001);
		assertEquals(0.5621765, output.get(2,0), 0.000001);
		assertEquals(0.5312094, output.get(3,0), 0.000001);
		assertEquals(0.5156199, output.get(4,0), 0.000001);
		assertEquals(0.5, output.get(5,0), 0.000001);
		assertEquals(0.2689414, output.get(6,0), 0.000001);
	}
}
