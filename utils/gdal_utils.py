from osgeo import gdal
import numpy as np

def get_img_shape(data_file:str)->tuple:
    '''
    通过gdal获取遥感图像的shape
    :param data_file:遥感图像路径
    :return:(weidgh, height, bands)
    '''
    dataset = gdal.Open(data_file, gdal.GA_ReadOnly) #只读方式打开
    weidth = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    img_shape = tuple([weidth, height, bands])

    return img_shape

def get_geo_transformer(data_file:str)->tuple:
    '''
    通过gdal获取遥感图像经纬度
    :param data_file:遥感图像路径
    :return:
    '''
    dataset = gdal.Open(data_file)
    if dataset is None:
        print("文件%s无法打开" % data_file)
        exit(-1)
    geo_transformer = dataset.GetGeoTransform()
    return geo_transformer

def read_img(data_file:str, width_offset:int, height_offset:int, width:int, height:int, band_idx=None)->np.array:
    '''

    :param data_file:
    :param width_offset:
    :param height_offset:
    :param width:
    :param height:
    :param band_idx:
    :return:
    '''
    dataset = gdal.Open(data_file, gdal.GA_ReadOnly)
    if dataset is None:
        print("文件%s无法打开" % data_file)
        exit(-1)

    img_width = dataset.RasterXSize
    img_height = dataset.RasterYSize
    if band_idx is None:
        img_bands = dataset.RasterCount
        band_idx =list(range(1, img_bands+1))
    else:
        img_bands = len(band_idx)

    #判断索引是否越界，只读取不越界的部分，其余部分补0
    block_width = width
    block_height = height
    if width_offset+width > img_width:
        block_width = img_width - width_offset
    if height_offset+height > img_height:
        block_height = img_height - height_offset
    img_data =np.zeros((img_bands, block_height, block_width), dtype=np.float)

    for i,idx in enumerate(band_idx):
        band =dataset.GetRasterBand(idx)
        img_data[i] =band.ReadAsArray(width_offset, height_offset, block_width, block_height)

    return img_data

def save_img(result_file:str, img:np.array, img_width:int, ing_height:int, img_bands:int, geoTransfor):
    '''

    :param result_file:
    :param img:
    :param img_width:
    :param ing_height:
    :param img_bansd:
    :param geoTransfor:
    :return:
    '''
    driver = gdal.GetDriverByName("GTIFF")
    dataset = driver.Create(result_file, img_width, ing_height, img_bands, gdal.GDT_Byte)
    if geoTransfor:
        dataset.SetGeoTransform(geoTransfor)
    for i in range(img_bands):
        dataset.GetRasterBand(i+1).WriteArray(img[i])



if __name__=="__main__":
    data_file = r'/data02/zht_vqa/change_detection/LearnGroup/S2A_MSIL2A_20200810T030551_N9999_R075_T50SLJ_m10.tiff'
    # img_shape = get_img_shape(data_file)
    # print(img_shape)
    # geo_trans = get_geo_transformer(data_file)
    # print(type(geo_trans))
    img = read_img(data_file, 1024, 1024, 512, 512, [3, 2, 1])
    print(img.shape)
    save_img('1.tiff', img, 512, 512, 3, None)
