package neural.network.exceptions;

import org.ejml.data.FMatrixRMaj;

public class InvalidInputException extends Exception {

	private static final long serialVersionUID = 426184486015582232L;

	private final FMatrixRMaj input;
	
	public InvalidInputException(final FMatrixRMaj input) {
		this.input = input;
	}
	
	public FMatrixRMaj getInput() {
		return input;
	}
	
}
