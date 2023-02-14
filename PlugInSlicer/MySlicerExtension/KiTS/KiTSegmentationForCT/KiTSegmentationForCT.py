import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import slicer
import logging
import subprocess
import tempfile
import numpy as np
import shutil
import time
#
# KiTSegmentationForCT
#

class KiTSegmentationForCT(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent = parent
    self.parent.title = "Kidney-Tumor Segmentation For CT"
    self.parent.categories = ["IMAG2"]
    self.parent.dependencies = []
    self.parent.contributors = ["Giammarco La Barbera (TelecomParis - IMAG2 Necker - Philips France)"]
    self.parent.helpText = """
This module performs automatic segmentation of Kidneys, Renal Tumor and (if possible) Ureters in CT images of children affected by Wilms' Tumor.
"""
    #self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This tool was developed for the IMAG2 lab. The AI method was developed thanks to: (i) the database of the DeePRAC project provided by the abdominal-visceral surgery department of Necker hospital; (ii) the grants from Region Ile de France (DIM RFSI) and Philips Research France. The methods used here were devised through a collaboration among (1) LTCI-Télecom Paris, (2) LIP6-Sorbonne Université, (3) IMAG2-Necker hospital and (4) Philips Research France."""

#
# KiTSegmentationForCTWidget
#

class KiTSegmentationForCTWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    self.reloadAndTestButton.setVisible(False)
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Selection and Segmentation"
    self.layout.addWidget(parametersCollapsibleButton)
    parametersCollapsibleButton1 = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton1.text = "Segment Editor"
    self.layout.addWidget(parametersCollapsibleButton1)


    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    parametersFormLayout1 = qt.QFormLayout(parametersCollapsibleButton1)


    self.parameters = slicer.vtkMRMLCropVolumeParametersNode()

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene(slicer.mrmlScene)
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    self.volume_node = self.inputSelector.currentNode()
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = False
    self.outputSelector.showHidden = False
    self.outputSelector.renameEnabled = True
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene(slicer.mrmlScene)
    self.outputSelector.setToolTip("Pick the output to the algorithm.")

    self.output_node = self.outputSelector.currentNode()
    parametersFormLayout.addRow("Output Segmentation: ", self.outputSelector)

    # Apply Button4
    #
    self.applyButton4 = qt.QPushButton("bones, liver and spleen")
    self.applyButton4.toolTip = "Run the algorithm for bones, liver and spleen vessels."
    self.applyButton4.enabled = False
    parametersFormLayout.addRow(self.applyButton4)

    # roi volume selector
    #
    self.roiSelector = slicer.qMRMLNodeComboBox()
    self.roiSelector.nodeTypes = ["vtkMRMLAnnotationROINode"]
    self.roiSelector.selectNodeUponCreation = True
    self.roiSelector.addEnabled = True
    self.roiSelector.removeEnabled = True
    self.roiSelector.noneEnabled = False
    self.roiSelector.showHidden = False
    self.roiSelector.renameEnabled = True
    self.roiSelector.showChildNodeTypes = False
    self.roiSelector.setMRMLScene(slicer.mrmlScene)
    self.roiSelector.setToolTip("Pick the input to the algorithm.")
    self.roi = self.roiSelector.currentNode()
    parametersFormLayout.addRow("AnnotationROI: ", self.roiSelector)

    #
    # Apply Button
    #
    self.fitButton = qt.QPushButton("Re-fit crop-cube to renal zone")
    self.fitButton.toolTip = "Run the algorithm."
    self.fitButton.enabled = False
    parametersFormLayout.addRow(self.fitButton)

    # preprocesed volume selector
    #
    self.prepinputSelector = slicer.qMRMLNodeComboBox()
    self.prepinputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.prepinputSelector.selectNodeUponCreation = True
    self.prepinputSelector.addEnabled = True
    self.prepinputSelector.removeEnabled = True
    self.prepinputSelector.noneEnabled = False
    self.prepinputSelector.showHidden = False
    self.prepinputSelector.renameEnabled = True
    self.prepinputSelector.showChildNodeTypes = False
    self.prepinputSelector.setMRMLScene(slicer.mrmlScene)
    self.prepinputSelector.setToolTip("Pick the preprocessed input to the algorithm.")
    self.prepvolume_node = self.prepinputSelector.currentNode()
    parametersFormLayout.addRow("Preprocessed Input Volume: ", self.prepinputSelector)


    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("cropping and pre-processing")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    line = qt.QFrame()
    line.setFrameShape(qt.QFrame().HLine)
    line.setFrameShadow(qt.QFrame().Sunken)
    line.setStyleSheet("min-height: 24px")
    parametersFormLayout.addRow(line)

    text = qt.QLabel()
    text.setText("It is suggested to do (cropping and pre-processing) first")
    parametersFormLayout.addRow(text)
    text = qt.QLabel()
    text.setText("so that automatic segmentation is more performant in accuracy and time.")
    parametersFormLayout.addRow(text)


    #
    # Apply Button1
    #
    self.applyButton1 = qt.QPushButton("kidneys and renal masses")
    self.applyButton1.toolTip = "Run the algorithm for kidneys and renal masses."
    self.applyButton1.enabled = False
    parametersFormLayout.addRow(self.applyButton1)

    # Apply Button2
    #
    self.applyButton2 = qt.QPushButton("ureters")
    self.applyButton2.toolTip = "Run the algorithm for ureters."
    self.applyButton2.enabled = False
    parametersFormLayout.addRow(self.applyButton2)

    # Apply Button3
    #
    self.applyButton3 = qt.QPushButton("blood vessels")
    self.applyButton3.toolTip = "Run the algorithm for blood vessels."
    self.applyButton3.enabled = False
    parametersFormLayout.addRow(self.applyButton3)

    # Segment Editor
    #

    self.segmentEditorWidget = slicer.modules.segmenteditor.createNewWidgetRepresentation()
    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    parametersFormLayout1.addWidget(self.segmentEditorWidget)
    #slicer.util.findChild(self.segmentEditorWidget, "Reload && Test").hide()

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.fitButton.connect('clicked(bool)', self.onFitButton)
    self.applyButton1.connect('clicked(bool)', self.onApplyButton1)
    self.applyButton2.connect('clicked(bool)', self.onApplyButton2)
    self.applyButton3.connect('clicked(bool)', self.onApplyButton3)
    self.applyButton4.connect('clicked(bool)', self.onApplyButton4)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onROISelect)
    self.prepinputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onOutputSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()
    self.onROISelect()
    self.onInputSelect()
    self.onOutputSelect()

    # Initialize
    self.a = None
    self.b = None

  def cleanup(self):
    pass

  def onSelect(self):
    self.volume_node = self.inputSelector.currentNode()

  def onOutputSelect(self):
    self.applyButton4.enabled = self.outputSelector.currentNode()
    self.output_node = self.outputSelector.currentNode()
    self.a = self.outputSelector.currentNode()
    if self.b:
      self.applyButton1.enabled = self.outputSelector.currentNode()
      self.applyButton2.enabled = self.outputSelector.currentNode()
      self.applyButton3.enabled = self.outputSelector.currentNode()




  def onROISelect(self):
    self.roi = self.roiSelector.currentNode()
    self.fitButton.enabled = self.roiSelector.currentNode()
    self.parameters.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(self.parameters)
    self.parameters.SetROINodeID(self.roi.GetID())
    self.parameters.SetInputVolumeNodeID(self.volume_node.GetID())
    logic = KiTSegmentationForCTLogic()
    print("Run Cropping Initialization")
    logic.run_crop(self.volume_node, self.roi)


  def onInputSelect(self):
    self.applyButton.enabled = self.prepinputSelector.currentNode()
    self.prepvolume_node = self.prepinputSelector.currentNode()
    self.b = self.prepinputSelector.currentNode()
    if self.a:
      self.applyButton1.enabled = self.outputSelector.currentNode()
      self.applyButton2.enabled = self.outputSelector.currentNode()
      self.applyButton3.enabled = self.outputSelector.currentNode()



  def onApplyButton(self):
      self.parameters.SetOutputVolumeNodeID(self.prepvolume_node.GetID())
      logic = KiTSegmentationForCTLogic()
      print("Run preprocessing")
      logic.run_prep(self.volume_node, self.prepvolume_node, self.parameters)

  def onFitButton(self):
      start_time = time.time()
      self.parameters.SetOutputVolumeNodeID(self.volume_node.GetID())
      s1 = time.time()
      print('set param',s1 - start_time)
      logic = KiTSegmentationForCTLogic()
      print("Run Cropping Initializzation")
      s2 = time.time()
      print('set logic',s2 - s1)
      logic.run_crop(self.volume_node, self.roi)
      end_time = time.time()
      print('total',end_time - start_time)

  def onApplyButton1(self):
      logic = KiTSegmentationForCTLogic()
      print("Run the algorithm")
      logic.run(self.volume_node, self.prepvolume_node, self.output_node, net=1)

  def onApplyButton2(self):
      logic = KiTSegmentationForCTLogic()
      print("Run the algorithm")
      logic.run(self.volume_node, self.prepvolume_node, self.output_node, net=2)

  def onApplyButton3(self):
    logic = KiTSegmentationForCTLogic()
    print("Run the algorithm")
    logic.run(self.volume_node, self.prepvolume_node, self.output_node, net=3)

  def onApplyButton4(self):
    logic = KiTSegmentationForCTLogic()
    print("Run the algorithm")
    logic.run_others(self.volume_node, self.prepvolume_node, self.output_node)

# KiTSegmentationForCTLogic
#

class KiTSegmentationForCTLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  def run_crop(self,originalVolume,roiNode):
    s3 = time.time()
    logging.info('Fit crop in renal zone')
    logging.info('Waiting...')
    Original = originalVolume.GetID()
    s4 = time.time()
    print('load original',s4 - s3)
    with tempfile.TemporaryDirectory() as dirpath:
      s5 = time.time()
      print('create tmp',s5 - s4)
      orig = slicer.mrmlScene.GetNodeByID(Original)
      s56 = time.time()
      print('get node from ID',s56 - s5)
      slicer.util.saveNode(orig, os.path.join(dirpath, 'original.nii'))
      s6 = time.time()
      print('save original',s6 - s5)
      startupinfo = None
      if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
      s7 = time.time()
      print('set up docker',s7 - s6)
      p = subprocess.Popen(['docker','run','--gpus','all','-it','--name','kits_infer','--rm','-v',dirpath+':/kits_build/data','kits:latest','--crop=True'], startupinfo=startupinfo)
      print("Running...")
      p.communicate()
      s8 = time.time()
      print('real running',s8 - s7)
      theta = np.load(os.path.join(dirpath, 'theta.npz'), allow_pickle=True)
      scaling = theta['scaling']
      tr = theta['translation']
      affine = theta['affine']
    # get the size of the volume voxels
    spacing = originalVolume.GetSpacing()
    print(spacing)
    print(affine)
    translation = np.matmul(affine,np.append(tr,1))
    roiNode.SetXYZ(translation[0], translation[1], translation[2])
    roiNode.SetRadiusXYZ(scaling[0]* spacing[0], scaling[1]* spacing[1], scaling[2] * spacing[2])
    roiNode.Initialize(slicer.mrmlScene)
    s9 = time.time()
    print('create annotation',s9 - s8)



  def run_prep(self, originalVolume, inputVolume, parameters):
    """
    Run the actual algorithm
    """

    logging.info('Preprocessing started')
    logging.info('Waiting...')
    slicer.modules.cropvolume.logic().Apply(parameters)
    slicer.util.setSliceViewerLayers(background=inputVolume)


    Input = inputVolume.GetID()
    Original = originalVolume.GetID()
    with tempfile.TemporaryDirectory() as dirpath:
      slicer.util.saveNode(slicer.mrmlScene.GetNodeByID(Input), os.path.join(dirpath, 'image.nii'))
      slicer.util.saveNode(slicer.mrmlScene.GetNodeByID(Original), os.path.join(dirpath, 'original.nii'))
      startupinfo = None
      if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
      p = subprocess.Popen(['docker','run','--gpus','all','-it','--name','kits_infer','--rm','-v',dirpath+':/kits_build/data','kits:latest','--prep=True'], startupinfo=startupinfo)
      print("Running...")
      p.communicate()
      shutil.copyfile(os.path.join(dirpath, 'prep.npz'), os.path.join(slicer.app.temporaryPath, 'prep.npz'))

    logging.info('Preprocessing completed')

  def run(self, originalVolume, inputVolume, outputVolume, net):
    """
    Run the actual algorithm
    """

    logging.info('Processing started')
    logging.info('Waiting...')


    with tempfile.TemporaryDirectory() as dirpath:
      if os.path.exists(os.path.join(slicer.app.temporaryPath, 'prep.npz')):
        shutil.copyfile(os.path.join(slicer.app.temporaryPath, 'prep.npz'), os.path.join(dirpath, 'prep.npz'))
      startupinfo = None
      if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
      p = subprocess.Popen(['docker','run','--gpus','all','-it','--name','kits_infer','--rm','-v',dirpath+':/kits_build/data','kits:latest','--net='+str(net)], startupinfo=startupinfo)
      print("Running...")
      p.communicate()
      n0 = outputVolume.GetSegmentation().GetNumberOfSegments()
      new_node = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_image.nii'))

      #a = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_0.nii'))
      #b = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_1.nii'))
      #c = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_2.nii'))
      #d = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_gauss.nii.gz'))

      ncs = np.load(os.path.join(dirpath, 'ncs.npy'))

    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(new_node, outputVolume)

    #slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(a, outputVolume)
    #slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(b, outputVolume)
    #slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(c, outputVolume)
    #slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(d, outputVolume)

    outputVolume.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
    outputVolume.CreateClosedSurfaceRepresentation()
    total = outputVolume.GetSegmentation().GetNumberOfSegments()
    k = 0
    if net==1:
      for segmentIndex in range(n0,total):
        k += 1
        if (segmentIndex-n0) < ncs[0]:
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Kidney_"+str(k))
          color = (0.7, 0.4, 0.3)
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
        elif (segmentIndex-n0) < (ncs[0]+ncs[1]):
          if (segmentIndex-n0) == ncs[0]:
            k = 1
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Mass_"+str(k))
          color = (0.5, 0.9, 0.5)
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
    elif net==2:
      for segmentIndex in range(n0,total):
        k += 1
        if (segmentIndex-n0) < ncs[0]:
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Ureter_"+str(k))
          color = (0.9, 0.7, 0.6)
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
    elif net==3:
      for segmentIndex in range(n0,total):
        if segmentIndex==n0:
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Arteries")
          color = (0.8, 0.4, 0.3)
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
        else:
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Veins")
          color = (0.0, 0.6, 0.8)
          outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)

    slicer.util.setSliceViewerLayers(background=originalVolume, foreground=outputVolume)
    slicer.mrmlScene.RemoveNode(new_node)

    logging.info('Processing completed')

  def run_others(self, originalVolume, inputVolume, outputVolume):
      """
      Run the actual algorithm
      """

      logging.info('Processing started')
      logging.info('Waiting...')

      Original = originalVolume.GetID()

      with tempfile.TemporaryDirectory() as dirpath:
        slicer.util.saveNode(slicer.mrmlScene.GetNodeByID(Original), os.path.join(dirpath, 'original.nii'))
        for net in range(4, 7):
          startupinfo = None
          if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
          p = subprocess.Popen(
            ['docker', 'run', '--gpus', 'all', '-it', '--name', 'kits_infer', '--rm', '-v', dirpath + ':/kits_build/data',
             'kits:latest', '--net=' + str(net)], startupinfo=startupinfo)
          print("Running...")
          p.communicate()
          n0 = outputVolume.GetSegmentation().GetNumberOfSegments()
          new_node = slicer.util.loadLabelVolume(os.path.join(dirpath, 'pred_image.nii'))
          slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(new_node, outputVolume)
          outputVolume.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)
          outputVolume.CreateClosedSurfaceRepresentation()
          if net == 4:
            segmentIndex = n0
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Bones")
            color = (0.9, 0.8, 0.5)
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
          elif net == 5:
            segmentIndex = n0
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Liver")
            color = (0.8, 0.5, 0.4)
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)
          else:
            segmentIndex = n0
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetName("Spleen")
            color = (0.6, 0.4, 0.6)
            outputVolume.GetSegmentation().GetNthSegment(segmentIndex).SetColor(color)

          slicer.util.setSliceViewerLayers(background=originalVolume, foreground=outputVolume)
          slicer.mrmlScene.RemoveNode(new_node)

      logging.info('Processing completed')




