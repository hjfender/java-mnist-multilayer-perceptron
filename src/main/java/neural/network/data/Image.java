package neural.network.data;

public class Image {

	private final int label;
	private final int[] data;
	
	public Image(final int label) {
		this.label = label;
		this.data = new int[784];
	}
	
	public void setPx(final int index, final int value) {
		this.data[index] = value;
	}
	
	public int getLabel() {
		return label;
	}
	
	public int[] getData() {
		return data;
	}
	
}
