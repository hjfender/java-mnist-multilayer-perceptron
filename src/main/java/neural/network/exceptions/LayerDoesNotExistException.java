package neural.network.exceptions;

public class LayerDoesNotExistException extends Exception {

	private static final long serialVersionUID = 3360224417568329720L;
	
	private final int layerNumber;
	
	public LayerDoesNotExistException(final int layerNumber) {
		super();
		this.layerNumber = layerNumber;
	}
	
	public int getLayerNumber() {
		return layerNumber;
	}
}
