'''
Utilities for reading/writing point cloud data to files
currentlty supported: 
    .vtp (efficient)
    .csv (simple and common)
'''
from typing import Union
from pandas import read_csv, DataFrame
from numpy import ndarray, array
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkPolyData 
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def readVtp(filename: str) -> vtkPolyData:
    '''
    Load .vtp file to vtk data.

    Parameters
    ----------
    filename: str
        .vtp file to be read.

    Returns
    -------
    data
        vtk polydata object.
    '''
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def readVtpToNumpy(filename: str) -> ndarray:
    '''
    Load .vtp point cloud to numpy array.

    Parameters
    ----------
    filename: str
        .vtp file to be read.

    Returns
    -------
    array
        numpy array of point cloud.
    '''
    out = readVtp(filename)
    return vtk_to_numpy(out.GetPoints().GetData())

def readCsvToNumpy(filename: str) -> ndarray:
    '''
    Load .csv point cloud to a numpy array.

    Parameters
    ----------
    filename: str
        .csv file to be read.

    Returns
    -------
    data: ndarray 
        array of point cloud.
    ''' 
    return read_csv(filename).to_numpy()

def readCsv(filename: str) -> vtkPolyData:
    '''
    Load .csv point cloud to vtkPolyData.

    Parameters
    ----------
    filename: str
        .csv file to be read.

    Returns
    -------
    data: vtkPolyData
        vtk polydata object.
    '''
    data = vtkPolyData()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(readCsvToNumpy(filename)))
    data.SetPoints(points)
    return data

def writeToVtp(data: Union[vtkPolyData, ndarray], filename: str) -> None:
    '''
    Write polydata to a .vtp file.

    Parameters
    ----------
    data: [vtkPolyData, ndarray]
        vtkPolyData object to write.
    filename: str
        output file name.
    '''
    if isinstance(data, ndarray):
        points = vtkPoints()
        points.SetData(numpy_to_vtk(data))
        data = vtkPolyData()
        data.SetPoints(points)

    writer = vtkXMLPolyDataWriter()
    writer.SetInputData(data)
    writer.SetFileName(filename)
    writer.Write()

def writeToCsv(data: Union[vtkPolyData, ndarray], filename: str) -> None:
    '''
    Write polydata to a .csv file.

    Parameters
    ----------
    data: [vtkPolyData, ndarray]
        vtkPolyData object to write.
    filename: str
        output file name.
    '''
    cols = ['x','y','z']
    if isinstance(data, vtkPolyData):
        data = vtk_to_numpy(data.GetPoints().GetData())
    df = DataFrame(data=data, columns=cols)# pyright: ignore
    df.to_csv(filename, index=False)
