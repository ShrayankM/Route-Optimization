import glob
import gdal
from scipy.ndimage import zoom

def combine(in_dir, out_dir):
    files = glob.glob(in_dir + '*')
    count = 0
    name = out_dir + 'Combined'
    fz = len(files)

    for file in files:

        f = gdal.Open(file, gdal.GA_ReadOnly)
        projection = f.GetProjection()
        geotransform = f.GetGeoTransform()

        f = f.ReadAsArray()
        fx, fy = f.shape

        count = count + 1
        if count == 1:
            outdata = gdal.GetDriverByName('GTiff').Create(name + '.tif', fy, fx, fz,gdal.GDT_UInt16)
            outdata.SetGeoTransform(geotransform)
            outdata.SetProjection(projection)
        outdata.GetRasterBand(count).WriteArray(f)
        outdata.FlushCache() 
    outdata = None

if __name__ == '__main__':
    input_dir = '/home/shrayank_mistry/Modules/Downscale to 10m/sen2to10/Output/'
    output_dir = '/home/shrayank_mistry/Modules/Downscale to 10m/sen2to10/'

    combine(input_dir, output_dir)