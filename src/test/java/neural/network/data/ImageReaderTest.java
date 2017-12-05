package neural.network.data;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.List;

import org.junit.Test;

public class ImageReaderTest {
	
	@Test
	public void testImageRead() throws IOException {
		List<Image> dataset = ImageReader.read("/mnist_test.csv");
		System.out.println(dataset.size());
		assertEquals(76, dataset.get(47).getData()[65]);
		assertEquals(174, dataset.get(47).getData()[66]);
		assertEquals(2, dataset.get(66).getData()[67]);
	}
}
