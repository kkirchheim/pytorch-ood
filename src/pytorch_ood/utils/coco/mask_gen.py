import json
import numpy as np
from PIL import Image, ImageDraw


def read_coco_annotations(annotations_file):
    """
    Load COCO annotations from a JSON file.

    :param annotations_file: Path to the COCO annotations JSON file.
    :return: Parsed COCO data. (dict)
    """
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)
    return coco_data


def rle_decode(rle, img_shape):
    """
    Decode RLE encoded mask into a binary mask.

    :param rle: Dictionary with 'counts' and 'size' for RLE encoding.
    :param img_shape: Tuple (height, width) of the image size.
    :return: Binary mask as a numpy array. (np.ndarray)
    """
    height, width = img_shape
    mask = np.zeros(height * width, dtype=np.uint8)

    counts = rle["counts"]
    size = rle["size"]

    if not size or size[0] != width or size[1] != height:
        print(
            f"Warning: RLE size {size} does not match the provided image dimensions {img_shape}."
        )
        raise ValueError("RLE size does not match the provided image dimensions.")

    # Convert counts to a flat mask array
    rle_array = np.array(counts, dtype=np.uint32)
    positions = np.concatenate(
        [
            np.arange(start, start + length)
            for start, length in zip(np.cumsum(rle_array[:-1]), rle_array[1:])
        ]
    )

    mask[positions] = 1
    mask = mask.reshape((height, width))

    return mask


def create_mask_from_segmentation(segmentation, image_size):
    """
    Create a binary mask from segmentation data, which can be either polygons or RLE.

    :param segmentation: List of polygons or RLE data representing the segmentation. (list of lists or dict)
    :param image_size: Size of the image as (width, height). (tuple of int)
    :return: Binary mask as a numpy array. (np.ndarray)
    """
    if isinstance(segmentation, dict):
        # Handle RLE segmentation
        if "counts" in segmentation and "size" in segmentation:
            img_shape = (image_size[1], image_size[0])  # (height, width)
            mask = rle_decode(segmentation, img_shape)
            mask = Image.fromarray(mask * 255)  # Convert binary mask to Image format
        else:
            raise ValueError("Unexpected RLE format.")
    else:
        # Handle polygon segmentation
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for polygon in segmentation:
            # Ensure polygon coordinates are in the correct format
            if isinstance(polygon, list) and all(isinstance(p, (int, float)) for p in polygon):
                if len(polygon) % 2 != 0:
                    raise ValueError(
                        "Polygon coordinates list should have an even number of elements."
                    )
                polygon = np.array(polygon).reshape(-1, 2).astype(int)
                polygon = [tuple(p) for p in polygon]
                draw.polygon(polygon, outline=1, fill=1)
            else:
                raise ValueError("Unexpected format for polygon coordinates.")

        mask = np.array(mask)

    return np.array(mask)


def generate_masks(coco_data, image_id):
    """
    Generate masks for a specific image ID from COCO annotations.

    :param coco_data: Parsed COCO data. (dict)
    :param image_id: ID of the image to generate masks for. (int)
    :return: List of binary masks as numpy arrays. (list of np.ndarray)
    """
    print(f"Image ID: {image_id}")
    masks = []
    annotations = coco_data["annotations"]
    image_info = next(item for item in coco_data["images"] if item["id"] == image_id)
    image_size = (image_info["width"], image_info["height"])

    for annotation in annotations:
        if annotation["image_id"] == image_id:
            segmentation = annotation["segmentation"]
            mask = create_mask_from_segmentation(segmentation, image_size)
            masks.append(mask)

    return masks


# Example usage:
# annotations_file = 'path_to_coco_annotations.json'
# image_id = 1  # Change this to the ID of the image you want to process

# coco_data = read_coco_annotations(annotations_file)
# masks = generate_masks(coco_data, image_id)

# # You can now access the masks as numpy arrays.
# for i, mask in enumerate(masks):
#     print(f"Mask {i+1}:")
#     print(mask)
