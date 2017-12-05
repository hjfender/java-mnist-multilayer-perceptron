package neural.network;

import static org.junit.Assert.assertEquals;

import org.ejml.data.FMatrixRMaj;
import org.junit.Test;

public class CostDerivativeTest {
	
	@Test
	public void testCostDevivative() {
		System.out.println("Testing Cost Derivative...\n");
		float data1[][] = {{1.0f}, {1.0f}, {1.0f}, {1.0f}, {1.0f}, {1.0f}, {1.0f}};
		float data2[][] = {{1.0f}, {0.5f}, {0.25f}, {0.125f}, {0.0625f}, {0.0f}, {-1.0f}};
		FMatrixRMaj v1 = new FMatrixRMaj(data1);
		FMatrixRMaj v2 = new FMatrixRMaj(data2);
		FMatrixRMaj v3 = Operations.cost_derivative(v1, v2);
		assertEquals(0.0f,v3.get(0, 0), 0.000001);
		assertEquals(0.5f,v3.get(1, 0), 0.000001);
		assertEquals(0.75f,v3.get(2, 0), 0.000001);
		assertEquals(0.875f,v3.get(3, 0), 0.000001);
		assertEquals(0.9375f,v3.get(4, 0), 0.000001);
		assertEquals(1.0f,v3.get(5, 0), 0.000001);
		assertEquals(2.0f,v3.get(6, 0), 0.000001);
	}
}
