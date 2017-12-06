package neural.network;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import neural.network.data.Image;
import neural.network.data.ImageReader;
import neural.network.exceptions.InvalidInputException;
import neural.network.exceptions.LayerDoesNotExistException;
import neural.network.exceptions.LayerTooSmallException;
import neural.network.exceptions.NetworkTooSmallException;

public class Run {

	public static void main(String[] args) throws NetworkTooSmallException, LayerTooSmallException, IOException, InvalidInputException, LayerDoesNotExistException {
		Network network = new Network(Arrays.asList(new Integer(784), new Integer(30), new Integer(10)));
		List<Image> testData = ImageReader.read("/mnist_test.csv");
		Operations.evaluate(testData, network);
		System.out.println("Initial evaluation: " + Operations.evaluate(testData, network) + " / " + testData.size());
		Operations.stochasticGradientDescent(network, 30, 10, 3.0f, true);
	}
}
