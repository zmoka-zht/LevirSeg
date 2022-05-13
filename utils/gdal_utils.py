from osgeo import gdal
import numpy as np

def get_img_shape(data_file:str)->tuple:
    '''
    通过gdal获取遥感图像的shape
    :param data_file:遥感图像路径
    :return:(weidgh, height, bands)
    '''
    dataset = gdal.Open(data_file, gdal.GA_ReadOnly) #只读方式打开
    weight = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    img_shape = tuple([weight, height, bands])

    return img_shape

if __name__=="__main__":
    data_file = r'/data02/zht_vqa/change_detection/LearnGroup/S2A_MSIL2A_20200810T030551_N9999_R075_T50SLJ_m10.tiff'
    img_shape = get_img_shape(data_file)
    print(img_shape)
