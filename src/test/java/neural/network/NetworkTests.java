package neural.network;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public class NetworkTests {

	@Test
	public void testInitializationOfNetwork() {
		List<Integer> sizes = Arrays.asList(new Integer(3), new Integer(2));
		Network network = new Network(sizes);
		System.out.println(network);
	}
}
