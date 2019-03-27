import multiresolutionimageinterface as mir
import cv2
import numpy as np
import os
import zipfile
from numpy.lib.stride_tricks import as_strided
from itertools import product
import numbers
from sklearn.utils import check_array, check_random_state
from sklearn.base import BaseEstimator

reader = mir.MultiResolutionImageReader()
writer = mir.MultiResolutionImageWriter()

LEVEL = 4  # This is the current downsample level

# Takes in the desired patch width and height for the Whole Slide image
# and tiles them into smaller image patches using the thresholded mask
# Here there is 50% overlap
def extract_wsi_patches_sliding(pw, ph, img, otsu_mask, tumor_mask, thresh, patient_node, patch_dir, isNormal):
    if (not os.path.exists(patch_dir)):
        os.makedirs(patch_dir)
    if (not os.path.exists(os.path.join(patch_dir, '1'))):
        os.makedirs(os.path.join(patch_dir, '1'))
    if (not os.path.exists(os.path.join(patch_dir, '0'))):
        os.makedirs(os.path.join(patch_dir, '0'))
    patch_count = 0
    # Loop over the image and look at the overlapping regions when there are tumor
    # cells
    mask0_written = False
    i = 0
    while (i < (img.shape[0] - pw)):
        x = int(i)
        j = 0
        while (j < (img.shape[1] - ph)):
            y = int(j)
            j += ph/2
            sub_image = otsu_mask[x:x+pw, y:y+ph]
            nz_count = np.where(sub_image > thresh)
            db_key = patient_node+'_'+str(patch_count) + '_'+str(x)+'_'+str(y)
            if (not isNormal):
                tumor_sub_image = tumor_mask[x:x+pw, y:y+ph,:]
                tumor_count = np.where(tumor_sub_image > 0)
                if ((nz_count[0].shape[0]/(pw*ph)) > 0.8):

                    # Encoding the string for label as the:
                    # patient_node + patch count + tumor or not label
                    # Checking that if there is any tumor cell in the center region
                    # of 128x128
                    tx = tumor_count[0]
                    ty = tumor_count[1]
                    txa = np.greater_equal(tx, 64)*np.less_equal(tx, 192)
                    tya = np.greater_equal(ty, 64)*np.less_equal(ty, 192)
                    if (tumor_count[0].shape[0]>0):
                        if (np.dot(txa, tya)):
                            patch_file = db_key+'_C.png'
                        else:
                            patch_file = db_key+'.png'
                        patch_tmp = os.path.join(patch_dir, '1')
                        mask_file = db_key + '_Mask.png'
                        cv2.imwrite(os.path.join(patch_tmp, mask_file), tumor_sub_image)
                        cv2.imwrite(os.path.join(patch_tmp, patch_file), img[x:x+pw,y:y+ph,:])
                    else:
                        patch_file = db_key+'.png'
                        patch_tmp = os.path.join(patch_dir, '0')
                        if (not mask0_written):
                            mask_file = db_key + '_Mask.png'
                            cv2.imwrite(os.path.join(patch_tmp, mask_file), tumor_sub_image)
                            mask0_written = True
                        cv2.imwrite(os.path.join(patch_tmp, patch_file), img[x:x+pw,y:y+ph,:])
                    patch_count = patch_count+1
            else:
                if ((nz_count[0].shape[0]/(pw*ph)) > 0.8):
                    print ("Writing file ", db_key)
                    patch_file = db_key+'.png'
                    patch_tmp = os.path.join(patch_dir, '0')
                    cv2.imwrite(os.path.join(patch_tmp, patch_file), img[x:x+pw,y:y+ph,:])
                    patch_count = patch_count+1
        i += pw/2 

    return patch_count


# The following functions are modified Scikit-learn implementation for
# image processing utilities
def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_shape_strides(arr, patch_shape=8, extraction_step=1):
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
        print ("Patch shape", patch_shape)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)
        print ("extraction step", extraction_step)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))
    return shape, strides

def extract_patches(arr, shape, strides):
    return as_strided(arr, shape=shape, strides=strides)


def extract_patches_2d(image, otsu_mask, tumor_mask, patch_size,
         thresh, patient_node, extraction_step=1, max_patches=None):
    i_h, i_w = image.shape[:2]
    print ("Dimensions are (w, h): ", i_w, i_h)
    p_h, p_w = patch_size
    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    # This creates strided one extraction patch
    img_shape, img_strides = extract_shape_strides(image,patch_shape=(p_h, p_w, n_colors), extraction_step=extraction_step)
    extracted_img_patches = extract_patches(image, img_shape, img_strides)
    extracted_otsu_patches = extract_patches(otsu_mask, img_shape, img_strides)

    if (isNormal):
        extracted_tumor_patches = extract_patches(tumor_mask, img_shape, img_strides)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    print ("extracted otsu patches ", extracted_otsu_patches.shape)
    if max_patches:
        random_set = set()
        patch_count = 0
        rng = check_random_state(0)
        nz_count = np.where(otsu_mask > thresh)
        print (nz_count[0].shape[0])
        while (patch_count < max_patches):
            idx = rng.randint(nz_count[0].shape[0], size=1)[0]
            if (idx in random_set):
                continue
            x = nz_count[0][idx]
            y = nz_count[1][idx]
            # print ("x and y ", x, y)
            patch_file = patient_node + '_' + str(x) + '_' + str(y) + '.png'
            if ((x >= extracted_otsu_patches.shape[0] ) or (y >= extracted_otsu_patches.shape[1])):
                continue
            otsu_patch = extracted_otsu_patches[x, y, 0]
            tmp = np.where(otsu_patch > thresh)

            print (tmp[0].shape[0]/(p_w*p_h))
            if (tmp[0].shape[0]/(p_w*p_h) > 0.4):
                random_set.insert(idx)
                patch_count += 1
                if (isNormal):
                    print ("writing with patch count, file ", patch_count, patch_file)
                    patch_tmp = os.path.join(patch_dir, '0')
                    cv2.imwrite(os.path.join(patch_tmp, patch_file), extracted_img_patches[x, y, 0])
                else:
                    i_s = rng.randint(i_h - p_h + 1, size=n_patches)
                    j_s = rng.randint(i_w - p_w + 1, size=n_patches)
                    patches = extracted_patches[i_s, j_s, 0]
                    sub_image = otsu_mask[x:x+pw, y:y+ph]
                    tumor_sub_image = tumor_mask[x:x+p_w, y:y+p_h,:]
                    tumor_count = np.where(tumor_sub_image > 0)

                    # The number of tumor cells are less than max_patches
                    if (tumor_count[0].shape[0] < max_patches):
                        print ("Number of tumor cell are less than max ", max_patches)
                        return []
                    db_key = patient_node+'_'+str(patch_count) + '_0'+'_'+str(x)+'_'+str(y)
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    return patches

def extract_wsi_patches_random(pw, ph, img, otsu_mask, tumor_mask, thresh, patient_node):
    patches = extract_patches_2d(img, otsu_mask, tumor_mask, (pw, ph), thresh, patient_node, extraction_step=1,
            max_patches=1000)
    

def reconstruct_from_patches_2d(patches, image_size):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img

def otsu_thresholding(img):
    img_mask = np.zeros((img.shape[0], img.shape[1]))

    # Loop over all the  channels of the image
    thresh = 255
    for i in range(img.shape[2]):
        blur = cv2.GaussianBlur(img[:,:,i],(5,5),0)
        thresh_tmp, tmp = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img_mask = img_mask + tmp
        thresh = min(thresh, thresh_tmp)
    return thresh, (img_mask/float(img.shape[2]))

# Takes in the image file name of the TIF file, the annotation XML file
# Output path for the mask file, its stored in the path and with same
# name with _mask appended
def annotation_to_mask(img_name, xml_name, output_path):
    mr_image = reader.open(img_name)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_name)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    label_map = {'metastases': 1, 'normal': 2}
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map)
    return output_path

# Returns the patch from the MIR multi-resolution reader
# NOTE: It defaults to 0,0 to w,h of the image for a given level
# TODO: This can be extended to generate smaller patches and do
# otsu thresholding on individual patches.
def get_image(fname, level = 2):
    if (not os.path.exists(fname)):
        print ("file does not exist, ", fname)
        return
    mr_image = reader.open(fname)
    if (not mr_image):
        print ("file does not exist, ", fname)
        return

    numLevels = mr_image.getNumberOfLevels()
    w, h = (mr_image.getLevelDimensions(level))
    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(0 * ds), int(0 * ds), w, h, level)

    return mr_image, image_patch

def preprocess(img_fname, patient_node, isNormal, patch_dir, mask_path):
    mr_image, img = get_image(img_fname, LEVEL)
    print (img[:,:,0].shape)
    thresh, otsu_mask = otsu_thresholding(img)
    print (thresh)
    if (not isNormal):
        # Generate the mask file from the annotation list provided in the XML file
        mask_fname = os.path.join(mask_path, patient_node+'_Mask.tif')
        tm_image, tumor_mask = get_image(mask_fname, LEVEL)
        patch_list = extract_wsi_patches_sliding(256, 256, img, otsu_mask, tumor_mask,
                thresh, patient_node, patch_dir, isNormal)
    else:
        patch_list = extract_wsi_patches_sliding(256, 256, img, otsu_mask, None, thresh,
                patient_node, patch_dir, isNormal)
    return


def generate_patches(path, isNormal, patch_dir, mask_path):
    # Loops over all the XML file and extract corresponding .tif files and
    # generates patches from them
    for fname in os.listdir(path):
        if fname.endswith('.tif'):
            filename = os.path.splitext(fname)[0]
            img_path = os.path.join(path, filename+'.tif')
            preprocess(img_path, filename, isNormal, patch_dir, mask_path)


def main():
    training_str_normal = ['Z:\\Gdrive', 'CAMELYON16', 'TrainingData', 'Train_Normal']
    training_str_tumor = ['Z:\\Gdrive', 'CAMELYON16', 'TrainingData', 'Train_Tumor']
    test_path_str = ['Z:\\Gdrive', 'CAMELYON16', 'Testset', 'Images']
    mask_training_str = ['C:\\Users\\kulinseth', 'ml', 'cameylon2017','cameylon2016', 'TrainingData', 'Ground_Truth', 'Mask']
    mask_test_str = ['C:\\Users\\kulinseth', 'ml', 'cameylon2017','cameylon2016', 'Testset', 'Ground_Truth', 'Mask']
    patch_path_str = ['C:\\Users\\kulinseth', 'ml', 'cameylon2017']
    mask_path = os.path.join(*mask_training_str)
    patch_path = os.path.join(*patch_path_str)
    patch_dir = os.path.join(patch_path, 'patches_test')
    training_path_normal = os.path.join(*training_str_normal)
    training_path_tumor = os.path.join(*training_str_tumor)
    test_path = os.path.join(*test_path_str)
    generate_patches(test_path, True, patch_dir, mask_path)
    patch_dir = os.path.join(patch_path, 'patches_sliding')
    generate_patches(training_path_normal, True, patch_dir)
    # patch_dir = os.path.join(patch_path, 'patches_random')
    # generate_patches(training_path_normal, True, patch_dir)
if __name__ == '__main__':
    main()
