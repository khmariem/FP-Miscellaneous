import sys
import vtk
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(sys.argv[1])
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(sys.argv[2])
writer.SetInputConnection(reader.GetOutputPort())
writer.SetDataModeToAscii()
writer.Write()
