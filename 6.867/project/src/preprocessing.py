import multiresolutionimageinterface as mir
import cv2
import torch as th
import numpy as np
import os
import zipfile
import leveldb
import tables
import matplotlib.pyplot as plt
import subprocess

# HDF5 dataset format was used to store the data for the Camelyon dataset.
# LevelDB dataset format was also experimented with.
hdf5_path = './dataset.hdf5'
img_dtype = tables.UInt8Atom()
label_dtype = tables.StringAtom(itemsize=32)
hdf5_file = tables.open_file(hdf5_path, mode='w')
data_shape = (0, 256, 256, 3)

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
train_label = hdf5_file.create_earray(hdf5_file.root, 'train_label',
        label_dtype, shape=(0,))
val_label = hdf5_file.create_earray(hdf5_file.root, 'val_label', label_dtype,
        shape=(0, 100))
test_label = hdf5_file.create_earray(hdf5_file.root, 'test_label', label_dtype,
        shape=(0, 100))

train_file_id = {}
test_file_id = {}

# hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
# hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
# hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)

reader = mir.MultiResolutionImageReader()
writer = mir.MultiResolutionImageWriter()
training_path = os.path.join(os.getcwd(), 'training')

LEVEL = 4  # This is the current downsample level


# Takes in the desired patch width and height for the Whole Slide image
# and tiles them into smaller image patches using the thresholded mask
# The patches will be a list of dictionaries, with patch itself and
# patches = [patch1, patch2 ...]
# patch1 = {image:'img', is_tumor:{0,1}}
# Here there is 50% overlap
def create_wsi_patches(pw, ph, img, otsu_mask, tumor_mask, thresh, patient_node):
    patches = []
    patch = {"image": None, "is_tumor": 0}
    patch_dir = os.path.join(training_path, 'patches')
    patch_count = 0
    for i in range(int(img.shape[0]/pw)):
        for j in range(int(img.shape[1]/ph)):
            x = i*pw
            y = j*ph
            sub_image = otsu_mask[x:x+pw, y:y+ph]
            tumor_sub_image = tumor_mask[x:x+pw, y:y+ph,:]
            nz_count = np.where(sub_image > thresh)
            tumor_count = np.where(tumor_sub_image > 0)
            db_key = patient_node+'_'+str(patch_count) + '_0'+'_'+str(x)+'_'+str(y)
            if ((nz_count[0].shape[0]/(pw*ph)) > 0.8):

                # Encoding the string for label as the:
                # patient_node + patch count + tumor or not label
                if (tumor_count[0].shape[0]>1):
                    patch_file = db_key+'.png'
                    patch_tmp = os.path.join(patch_dir, '1')
                    cv2.imwrite(os.path.join(patch_tmp, patch_file), img[x:x+pw,y:y+ph,:])
                else:
                    patch_file = db_key+'.png'
                    patch_tmp = os.path.join(patch_dir, '0')
                    cv2.imwrite(os.path.join(patch_tmp, patch_file), img[x:x+pw,y:y+ph,:])
                train_storage.append(img[x:x+pw,y:y+ph,:][None])
                train_label.append(np.array([db_key], dtype='S32'))
                print (db_key)
                patch_count = patch_count+1
    return patch_count

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

def show_image(image):
    if 1:
        cv2.namedWindow("image")
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(image)

# Returns the patch from the MIR multi-resolution reader
# otsu thresholding on individual patches.
def get_image(fname, level = 2):
    if (not os.path.exists(fname)):
        return
    mr_image = reader.open(fname)
    numLevels = mr_image.getNumberOfLevels()
    w, h = (mr_image.getLevelDimensions(level))
    print ("Dimensions are (w, h): ", w, h)
    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(0 * ds), int(0 * ds), w, h, level)

    return mr_image, image_patch

def preprocess(img_fname, xml_fname, patient_node):
    mr_image, img = get_image(img_fname, LEVEL)
    print (img[:,:,0].shape)
    thresh, otsu_mask = otsu_thresholding(img)
    #plt.imshow(otsu_mask, cmap='gray')
    # Generate the mask file from the annotation list provided in the XML file
    mask_fname = os.path.join(training_path, patient_node+'_mask.tif')
    if (not os.path.exists(mask_fname)):
        annotation_to_mask(img_fname, xml_fname, mask_fname)
    tm_image, tumor_mask = get_image(mask_fname, LEVEL)
    patch_list = create_wsi_patches(256, 256, img, otsu_mask, tumor_mask,
            thresh, patient_node)

    # cleanup, delete the mask file
    os.remove(mask_fname)
    return

# This Queries the Google Drive shared folder for Camelyon dataset and downloads
# all the file-id's corresponding to the Patient data. The results were stored on
# a local Hard disk. As this was not enough, this approach was later extended to store
# all the results on the NAS storage.
def file_id():
    a = subprocess.check_output(["./gdrive", "list", "--query", "name contains 'patient'",
                        "--absolute", "--max",  "10000", "--no-header"])
    b = a.decode("utf-8")
    b = b.split('\n')
    for i in b:
        c = i.split(" ")
        c = list(filter(None, c))
        if c:
            name = c[1].split('/')
            patient_name = os.path.splitext(name[-1])[0]
            print (name)
            if 'training' in c[1]:
                train_file_id[patient_name] = c[0]
                print ('training', c[0], name[-1])
            elif 'testing' in c[1]:
                test_file_id[patient_name] = c[0]
                print ('testing', c[0], name[-1])

# This function loops over all the XML files with tumor annotations and downloads the
# file for patch generataion.
def generate_patches():
    # Loops over all the XML file and extract corresponding .tif files and
    # generates patches from them
    patient_duplicates = {}
    for xml_fname in os.listdir(training_path):
        if xml_fname.endswith('.xml'):
            filename = os.path.splitext(xml_fname)[0]
            patient_name = '_'.join(filename.split('_')[:2])
            if patient_name in patient_duplicates:
                patient_duplicates[patient_name] = True
                print ('duplicate ', patient_name)
            else:
                patient_duplicates[patient_name] = False

    for xml_fname in os.listdir(training_path):
        if xml_fname.endswith('.xml'):
            filename = os.path.splitext(xml_fname)[0]
            patient_name = '_'.join(filename.split('_')[:2])
            patient_found = False
            img_path = os.path.join(training_path, filename+'.tif')
            xml_path = os.path.join(training_path, xml_fname)
            # if patient == patient_name:
            for root, dirs, files in os.walk(training_path):
                for zfile in files:
                    if zfile.endswith('.zip') and (patient_name in zfile):
                        patient_found = True
                        zfile_path = os.path.join(root, zfile)
                        print ("Found the .zip file ", patient_name)
                        if 0:
                            if (not os.path.exists(img_path)):
                                with zipfile.ZipFile(zfile_path) as z:
                                    print ("Extracting ", filename)
                                    z.extract(filename+'.tif', training_path)
                                    preprocess(img_path, xml_path, filename)
                                    os.remove(img_path)
                            else:
                                preprocess(img_path, xml_path, filename)
                                os.remove(img_path)
            # If patient is not found locally on the Drive, then download it
            # from gdrive
            if (not patient_found):
                if patient_name in train_file_id:
                    zfile = os.path.join(os.getcwd(), patient_name+'.zip')
                    if (not os.path.exists(zfile)):
                        subprocess.check_output(['./gdrive', 'download',
                            train_file_id[patient_name]])
                    with zipfile.ZipFile(zfile) as z:
                        print("Extracting ", patient_name)
                        z.extract(filename+'.tif', training_path)
                        preprocess(img_path, xml_path, filename)
                        os.remove(img_path)
                        if (not patient_duplicates[patient_name]):
                            os.remove(zfile)

def main():
    file_id()
    generate_patches()
    hdf5_file.close()
if __name__ == '__main__':
    main()
