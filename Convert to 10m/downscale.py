import glob
import gdal
from scipy.ndimage import zoom

def setGeo(geotransform, bgx, bgy, x_offset = 0):
    if x_offset == 0:
        x_offset = geotransform[1]
        y_offset = geotransform[5]
    else:
        x_offset = x_offset
        y_offset = -x_offset
    reset0 = geotransform[0] + bgx * geotransform[1]
    reset3 = geotransform[3] + bgy * geotransform[5]
    reset = (reset0, x_offset, geotransform[2],
             reset3, geotransform[4], y_offset)
    return reset


def convert(file, out_dir):
    f = gdal.Open(file, gdal.GA_ReadOnly)
    projection = f.GetProjection()
    geotransform = f.GetGeoTransform()

    f = f.ReadAsArray()

    new_geo = setGeo(geotransform, 0, 0, x_offset = 10)
    f = zoom(f, [2,2], order = 0, mode = 'nearest')
    fx, fy = f.shape

    file_name = file[-11:-4]
    file_dir = out_dir + '/' + file_name
    outdata = gdal.GetDriverByName('GTiff').Create(file_dir +'.tif', fy, fx, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(new_geo)
    outdata.SetProjection(projection)
    outdata.GetRasterBand(1).WriteArray(f)
    outdata.FlushCache()
    outdata = None

def read_files(in_dir, out_dir):
    files = glob.glob(in_dir+'*')
    
    for f in files:
        convert(f, out_dir)


if __name__ == '__main__':
    input_dir = '/home/shrayank_mistry/Modules/Downscale to 10m/sen2to10/Input/'
    output_dir = '/home/shrayank_mistry/Modules/Downscale to 10m/sen2to10/Output/'

    read_files(input_dir, output_dir)