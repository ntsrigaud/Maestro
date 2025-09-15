# Vision Transformer (ViT) and Self-attention Mechanisms

## Processing Steps

1. Division of the input image into **patches** (non-overlapping small image tiles)
2. Normalization of the image channels of each patch from the 2D array
3. Processing of the input image by the transformer
4. **Flattening** of all normalized channels into one dimensional arrays of each patch in the image to obtain a **linear vector** for each patch
5. Conversion of the obtained pixel values into **embedding vectors** for each patch
   - Takes each input and pass it through a neural network
   - Allow them to be treated as **_word embeddings_**
6. Make three copies of each **embedding vector** to feed into:
   - Q: Query matrix
   - K: Key matrix
   - V: Value matrix

   to obtain the **_attention layer_**
   - Allows each embedding vectors to be processed in parallel

7. Addition of **_positional encoding_** to the **embedding vector**
   - Incorporates position information into the vector
8. Feed the final vector obtained to the **attention block**
9. Application of multiple **attention blocks**
10. Application of a neural network using **_[softmax](https://machinelearningtheory.org/docs/Unified-View/softmax-function/)_** for classification
11. Addition of an **_extra learnable embedding vector_**
    - Gather all important information from other patches using the **attention mechanism** for classification
12. Feed the obtained vector to the **classification layer** to obtain the **_softmax distribution_** for the class label

> [!IMPORTANT]
> The normal patch size when dividing the input image is 16x16. Increasing it to a higher value would prevent the transformer from gaining a good understanding of the patch.

> [!NOTE]
> By computing the _attention_ of a patch with all other patches, we can find out how this patch _relates_ to other patches. This helps the model _understand_ how different parts of the image are related to each other, enabling it to develop an overall understanding of the information in the image.

> [!NOTE]
> The attention mechanism in ViT gives it a **_global receptive field_**.

## CNNs vs ViT

### CNNs

- _Local_ receptive field
  - Built-in inductive bias
- May predict the class of an image using the **texture of the object**

### ViT

- _Global_ receptive field
  - Focused more on global features (shape) for classification

> [!IMPORTANT]
> ViT cannot compete with CNNs when there is a small amount of data, they require large datasets to perform well.

## References

- [Vision Tranformer](https://youtu.be/vJF3TBI8esQ?si=5P4VTtOqqwuNXAjk)
