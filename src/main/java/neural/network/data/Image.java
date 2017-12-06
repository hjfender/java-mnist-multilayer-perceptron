package neural.network.data;

import org.ejml.data.FMatrixRMaj;

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
	
	public FMatrixRMaj getVector() {
		FMatrixRMaj v = new FMatrixRMaj(data.length, 1);
		for(int i = 0; i < data.length; i++) {
			v.set(i, 0, data[i]);
		}
		return v;
	}
	
}
