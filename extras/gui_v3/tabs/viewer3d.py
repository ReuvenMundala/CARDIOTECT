from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSlider, QGroupBox, QSplitter, QFrame, QGridLayout)
from PyQt5.QtCore import Qt
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
import numpy as np

class Viewer3DTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = main_window.state
        
        # VTK Pipeline elements
        self.plane_widgets = []
        self.scene_3d_actors = []
        self.mappers_ct = []
        self.mappers_mask = []
        
        self.init_ui()
        
        # Listen for new data
        self.state.scan_loaded.connect(self.on_scan_loaded)
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = QLabel("Interactive 3D Anatomical Reconstruction")
        header.setObjectName("title")
        main_layout.addWidget(header)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Left Panel: Controls ---
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(20)
        
        # Tissue Opacity Control
        group_tissue = QGroupBox("Clinical Visualization")
        tissue_layout = QVBoxLayout(group_tissue)
        
        self.slider_tissue = QSlider(Qt.Horizontal)
        self.slider_tissue.setRange(0, 100)
        self.slider_tissue.setValue(25) # Default low opacity for tissue
        self.slider_tissue.valueChanged.connect(self.update_transfer_functions)
        
        tissue_layout.addWidget(QLabel("Heart Muscle Transparency:"))
        tissue_layout.addWidget(self.slider_tissue)
        left_layout.addWidget(group_tissue)
        
        # Calcium Color Control
        group_calc = QGroupBox("Diagnostic Overlays")
        calc_layout = QVBoxLayout(group_calc)
        
        self.btn_mode_raw = QPushButton("Raw Anatomical (White)")
        self.btn_mode_raw.setObjectName("secondary")
        self.btn_mode_raw.clicked.connect(lambda: self.set_calcium_mode('raw'))
        
        self.btn_mode_ai = QPushButton("AI Prediction (Glowing Red)")
        self.btn_mode_ai.setObjectName("primary")
        self.btn_mode_ai.clicked.connect(lambda: self.set_calcium_mode('ai'))
        
        self.btn_mode_gt = QPushButton("Ground Truth (Glowing Green)")
        self.btn_mode_gt.setObjectName("secondary")
        self.btn_mode_gt.clicked.connect(lambda: self.set_calcium_mode('gt'))
        
        calc_layout.addWidget(self.btn_mode_raw)
        calc_layout.addWidget(self.btn_mode_ai)
        calc_layout.addWidget(self.btn_mode_gt)
        left_layout.addWidget(group_calc)
        
        left_layout.addStretch()
        
        # --- Right Panel: 4-Panel Grid ---
        right_panel = QFrame()
        right_layout = QGridLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        self.vtk_ax = QVTKRenderWindowInteractor(self)
        self.vtk_cor = QVTKRenderWindowInteractor(self)
        self.vtk_sag = QVTKRenderWindowInteractor(self)
        self.vtk_3d = QVTKRenderWindowInteractor(self)
        
        self.slider_ax = QSlider(Qt.Horizontal)
        self.slider_cor = QSlider(Qt.Horizontal)
        self.slider_sag = QSlider(Qt.Horizontal)
        
        self.ren_ax = vtk.vtkRenderer()
        self.ren_cor = vtk.vtkRenderer()
        self.ren_sag = vtk.vtkRenderer()
        self.ren_3d = vtk.vtkRenderer()
        
        for ren, vtk_w in zip([self.ren_ax, self.ren_cor, self.ren_sag, self.ren_3d], 
                              [self.vtk_ax, self.vtk_cor, self.vtk_sag, self.vtk_3d]):
            ren.SetBackground(0.05, 0.02, 0.02)
            vtk_w.GetRenderWindow().AddRenderer(ren)
            
        self.vtk_3d.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        for vtk_w in [self.vtk_ax, self.vtk_cor, self.vtk_sag]:
            # Disable interaction for MPR views (only sliders control them)
            vtk_w.SetInteractorStyle(None)
            
        # Track maximization state
        self.maximized_panel = None
        
        def create_panel(vtk_w, slider, title, color, is_3d=False):
            frame = QFrame()
            frame.setStyleSheet(f"QFrame {{ border: 2px solid {color}; border-radius: 4px; background-color: #0D0505; }}")
            lay = QVBoxLayout(frame)
            lay.setContentsMargins(0,0,0,0)
            lay.setSpacing(0)
            
            # Header Container
            header_w = QWidget()
            header_w.setStyleSheet(f"background-color: {color}; border: none; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;")
            header_lay = QHBoxLayout(header_w)
            header_lay.setContentsMargins(8, 4, 8, 4)
            
            lbl = QLabel(title)
            lbl.setStyleSheet("color: white; font-weight: bold; border: none;")
            header_lay.addWidget(lbl)
            
            header_lay.addStretch()
            
            # 3D specific controls
            if is_3d:
                btn_reset = QPushButton("Reset View")
                btn_reset.setStyleSheet("background-color: rgba(255,255,255,0.2); color: white; border: none; padding: 2px 6px; border-radius: 2px;")
                btn_reset.setCursor(Qt.PointingHandCursor)
                btn_reset.clicked.connect(self.reset_3d_camera)
                header_lay.addWidget(btn_reset)
                
                self.btn_ortho = QPushButton("Ortho")
                self.btn_ortho.setCheckable(True)
                self.btn_ortho.setStyleSheet("""
                    QPushButton { background-color: rgba(255,255,255,0.2); color: white; border: none; padding: 2px 6px; border-radius: 2px; }
                    QPushButton:checked { background-color: rgba(255,255,255,0.6); color: black; font-weight: bold; }
                """)
                self.btn_ortho.setCursor(Qt.PointingHandCursor)
                self.btn_ortho.clicked.connect(self.toggle_ortho)
                header_lay.addWidget(self.btn_ortho)

            btn_max = QPushButton("[ ]")
            btn_max.setToolTip("Maximize / Restore")
            btn_max.setStyleSheet("background-color: transparent; color: white; font-weight: bold; border: none; padding: 0px;")
            btn_max.setCursor(Qt.PointingHandCursor)
            
            # Pass the frame itself to the toggle function
            btn_max.clicked.connect(lambda _, f=frame: self.toggle_maximize(f))
            header_lay.addWidget(btn_max)
            
            lay.addWidget(header_w)
            lay.addWidget(vtk_w, 1)
            if slider:
                lay.addWidget(slider)
                
            frame._title = title # save for identification
            return frame

        self.panel_ax = create_panel(self.vtk_ax, self.slider_ax, "Axial", "#8B0000")
        self.panel_cor = create_panel(self.vtk_cor, self.slider_cor, "Coronal", "#006400")
        self.panel_sag = create_panel(self.vtk_sag, self.slider_sag, "Sagittal", "#B8860B")
        self.panel_3d = create_panel(self.vtk_3d, None, "3D Heart", "#4A0E1B", is_3d=True)

        self.right_layout = right_layout
        self.right_layout.addWidget(self.panel_ax, 0, 0)
        self.right_layout.addWidget(self.panel_cor, 0, 1)
        self.right_layout.addWidget(self.panel_sag, 1, 0)
        self.right_layout.addWidget(self.panel_3d, 1, 1)
        
        # Add 3D-specific secondary sliders for maximized mode
        slider_3d_style = """
            QSlider::groove:horizontal { border-radius: 4px; height: 12px; background: #2D1515; }
            QSlider::handle:horizontal { background: %COLOR%; width: 24px; height: 24px; margin: -6px 0; border-radius: 12px; border: 2px solid white; }
        """
        self.slider_3d_ax = QSlider(Qt.Horizontal); self.slider_3d_ax.hide()
        self.slider_3d_ax.setStyleSheet(slider_3d_style.replace("%COLOR%", "#8B0000"))
        self.slider_3d_ax.setMinimumHeight(30)
        
        self.slider_3d_cor = QSlider(Qt.Horizontal); self.slider_3d_cor.hide()
        self.slider_3d_cor.setStyleSheet(slider_3d_style.replace("%COLOR%", "#006400"))
        self.slider_3d_cor.setMinimumHeight(30)
        
        self.slider_3d_sag = QSlider(Qt.Horizontal); self.slider_3d_sag.hide()
        self.slider_3d_sag.setStyleSheet(slider_3d_style.replace("%COLOR%", "#B8860B"))
        self.slider_3d_sag.setMinimumHeight(30)
        
        self.panel_3d.layout().addWidget(self.slider_3d_ax)
        self.panel_3d.layout().addWidget(self.slider_3d_cor)
        self.panel_3d.layout().addWidget(self.slider_3d_sag)
        self.panel_3d.layout().addSpacing(10) # Nudge from bottom edge
        
        # Sync secondary sliders with main sliders
        self.slider_ax.valueChanged.connect(self.slider_3d_ax.setValue)
        self.slider_3d_ax.valueChanged.connect(self.slider_ax.setValue)
        self.slider_cor.valueChanged.connect(self.slider_3d_cor.setValue)
        self.slider_3d_cor.valueChanged.connect(self.slider_cor.setValue)
        self.slider_sag.valueChanged.connect(self.slider_3d_sag.setValue)
        self.slider_3d_sag.valueChanged.connect(self.slider_sag.setValue)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800])
        main_layout.addWidget(splitter, 1)
        
        self.slider_ax.valueChanged.connect(self.on_slider_ax)
        self.slider_cor.valueChanged.connect(self.on_slider_cor)
        self.slider_sag.valueChanged.connect(self.on_slider_sag)

    def toggle_maximize(self, target_frame):
        panels = [self.panel_ax, self.panel_cor, self.panel_sag, self.panel_3d]
        if self.maximized_panel is None:
            # Hide all but target
            for p in panels:
                if p != target_frame:
                    p.setVisible(False)
            self.maximized_panel = target_frame
            # Show secondary sliders if 3D panel is maximized
            if target_frame == self.panel_3d:
                self.slider_3d_ax.show()
                self.slider_3d_cor.show()
                self.slider_3d_sag.show()
        else:
            # Restore all
            for p in panels:
                p.setVisible(True)
            self.maximized_panel = None
            # Hide secondary sliders
            self.slider_3d_ax.hide()
            self.slider_3d_cor.hide()
            self.slider_3d_sag.hide()
            
    def reset_3d_camera(self):
        if hasattr(self, 'ren_3d'):
            self.ren_3d.ResetCamera()
            self.vtk_3d.GetRenderWindow().Render()
            
    def toggle_ortho(self):
        if hasattr(self, 'ren_3d'):
            cam = self.ren_3d.GetActiveCamera()
            cam.SetParallelProjection(self.btn_ortho.isChecked())
            self.vtk_3d.GetRenderWindow().Render()

    def on_scan_loaded(self):
        """Called when Tab 2 finishes inference and loads the DICOM."""
        if self.state.vol_hu is None:
            return
        self.build_vtk_pipeline()

    def build_vtk_pipeline(self):
        """Builds the multi-actor surface pipeline and MPR planes."""
        for pw in self.plane_widgets:
            pw.Off()
        self.plane_widgets.clear()
        
        self.ren_ax.RemoveAllViewProps()
        self.ren_cor.RemoveAllViewProps()
        self.ren_sag.RemoveAllViewProps()
        self.ren_3d.RemoveAllViewProps()
        
        self.scene_3d_actors = []
        self.mappers_ct = []
        self.mappers_mask = []
        
        self.current_calc_mode = 'ai'
        self.update_volume_data()
        self._build_mpr_views()
        
    def _build_mpr_views(self):
        if self.state.vol_hu is None: return
        d, h, w = self.state.vol_hu.shape
        
        vtk_data_ct = numpy_support.numpy_to_vtk(num_array=self.state.vol_hu.astype(np.float32).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        self.img_ct = vtk.vtkImageData()
        self.img_ct.SetDimensions(w, h, d)
        if self.state.spacing:
            sz, sy, sx = self.state.spacing
            self.img_ct.SetSpacing(sx, sy, sz)
        self.img_ct.GetPointData().SetScalars(vtk_data_ct)
        
        self.img_mask = vtk.vtkImageData()
        self.img_mask.SetDimensions(w, h, d)
        if self.state.spacing:
            self.img_mask.SetSpacing(sx, sy, sz)
        vtk_data_mask = numpy_support.numpy_to_vtk(num_array=np.zeros(d*h*w, dtype=np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        self.img_mask.GetPointData().SetScalars(vtk_data_mask)
        
        colors = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
        orientations = [2, 1, 0] 
        renderers = [self.ren_ax, self.ren_cor, self.ren_sag]
        sliders = [self.slider_ax, self.slider_cor, self.slider_sag]
        
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        for i in range(3):
            orient = orientations[i]
            ren = renderers[i]
            slider = sliders[i]
            slider_3d = [self.slider_3d_ax, self.slider_3d_cor, self.slider_3d_sag][i]
            color = colors[i]
            
            # CT Mapper
            m_ct = vtk.vtkImageSliceMapper()
            m_ct.SetInputData(self.img_ct)
            m_ct.SetOrientation(orient)
            a_ct = vtk.vtkImageSlice()
            a_ct.SetMapper(m_ct)
            a_ct.GetProperty().SetColorWindow(1500)
            a_ct.GetProperty().SetColorLevel(300)
            ren.AddViewProp(a_ct)
            
            # Mask Mapper
            m_m = vtk.vtkImageSliceMapper()
            m_m.SetInputData(self.img_mask)
            m_m.SetOrientation(orient)
            a_m = vtk.vtkImageSlice()
            a_m.SetMapper(m_m)
            
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(5)
            lut.SetTableRange(0, 4)
            lut.SetTableValue(0, 0, 0, 0, 0)
            lut.SetTableValue(1, 1, 1, 0, 1) # Yellow (LM_LAD)
            lut.SetTableValue(2, 0, 0, 0, 0) # Disabled (Old LAD)
            lut.SetTableValue(3, 1, 0, 1, 1) # Magenta (LCX)
            lut.SetTableValue(4, 0, 0.5, 1, 1) # Blue (RCA) 
            lut.Build()
            a_m.GetProperty().SetLookupTable(lut)
            a_m.GetProperty().SetOpacity(0.5)
            a_m.GetProperty().SetUseLookupTableScalarRange(True)
            ren.AddViewProp(a_m)
            
            self.mappers_ct.append(m_ct)
            self.mappers_mask.append(m_m)
            
            # Camera
            cam = ren.GetActiveCamera()
            cam.ParallelProjectionOn()
            bounds = self.img_ct.GetBounds()
            cx, cy, cz = (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2
            dist = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 2
            
            if orient == 2:
                cam.SetPosition(cx, cy, cz + dist)
                cam.SetViewUp(0, -1, 0)
            elif orient == 1:
                cam.SetPosition(cx, cy - dist, cz)
                cam.SetViewUp(0, 0, 1)
            elif orient == 0:
                cam.SetPosition(cx + dist, cy, cz)
                cam.SetViewUp(0, 0, 1)
            cam.SetFocalPoint(cx, cy, cz)
            ren.ResetCamera()
            
            # Slider update
            slider.blockSignals(True)
            slider_3d.blockSignals(True)
            
            slider.setRange(0, self.img_ct.GetDimensions()[orient] - 1)
            slider_3d.setRange(0, self.img_ct.GetDimensions()[orient] - 1)
            
            val = self.img_ct.GetDimensions()[orient] // 2
            slider.setValue(val)
            slider_3d.setValue(val)
            
            slider.blockSignals(False)
            slider_3d.blockSignals(False)
            
            # 3D Plane
            pw = vtk.vtkImagePlaneWidget()
            pw.SetInteractor(self.vtk_3d.GetRenderWindow().GetInteractor())
            pw.SetPicker(picker)
            pw.RestrictPlaneToVolumeOn()
            pw.GetPlaneProperty().SetColor(color)
            pw.SetTexturePlaneProperty(vtk.vtkProperty())
            pw.SetResliceInterpolateToLinear()
            pw.SetInputData(self.img_ct)
            pw.SetPlaneOrientation(orient)
            pw.SetSliceIndex(self.img_ct.GetDimensions()[orient] // 2)
            pw.SetWindowLevel(1500, 300)
            pw.On()
            pw.InteractionOff() # Disable mouse dragging of MPR plane internally
            pw.AddObserver("InteractionEvent", self._create_plane_observer(i))
            self.plane_widgets.append(pw)
            
        self._update_mpr_overlays()
        self.on_slider_ax(self.slider_ax.value())
        self.on_slider_cor(self.slider_cor.value())
        self.on_slider_sag(self.slider_sag.value())
        
    def _create_plane_observer(self, index):
        def observer(obj, event):
            val = obj.GetSliceIndex()
            sliders = [self.slider_ax, self.slider_cor, self.slider_sag]
            sliders[index].blockSignals(True)
            sliders[index].setValue(val)
            sliders[index].blockSignals(False)
            
            self.mappers_ct[index].SetSliceNumber(val)
            self.mappers_mask[index].SetSliceNumber(val)
            
            renders = [self.vtk_ax, self.vtk_cor, self.vtk_sag]
            renders[index].GetRenderWindow().Render()
        return observer

    def on_slider_ax(self, val):
        if hasattr(self, 'mappers_ct') and self.mappers_ct:
            self.mappers_ct[0].SetSliceNumber(val)
            self.mappers_mask[0].SetSliceNumber(val)
            if len(self.plane_widgets) > 0: self.plane_widgets[0].SetSliceIndex(val)
            self.vtk_ax.GetRenderWindow().Render()
            self.vtk_3d.GetRenderWindow().Render()

    def on_slider_cor(self, val):
        if hasattr(self, 'mappers_ct') and self.mappers_ct:
            self.mappers_ct[1].SetSliceNumber(val)
            self.mappers_mask[1].SetSliceNumber(val)
            if len(self.plane_widgets) > 1: self.plane_widgets[1].SetSliceIndex(val)
            self.vtk_cor.GetRenderWindow().Render()
            self.vtk_3d.GetRenderWindow().Render()

    def on_slider_sag(self, val):
        if hasattr(self, 'mappers_ct') and self.mappers_ct:
            self.mappers_ct[2].SetSliceNumber(val)
            self.mappers_mask[2].SetSliceNumber(val)
            if len(self.plane_widgets) > 2: self.plane_widgets[2].SetSliceIndex(val)
            self.vtk_sag.GetRenderWindow().Render()
            self.vtk_3d.GetRenderWindow().Render()

    def _create_surface_actor(self, mask_np, color=(1.0, 1.0, 1.0), opacity=1.0, smoothing=True):
        if mask_np is None or np.sum(mask_np) == 0: return None
        d, h, w = mask_np.shape
        mask_float = mask_np.astype(np.float32)
        vtk_data = numpy_support.numpy_to_vtk(num_array=mask_float.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(w, h, d)
        if self.state.spacing:
            sz, sy, sx = self.state.spacing
            image_data.SetSpacing(sx, sy, sz)
        image_data.GetPointData().SetScalars(vtk_data)
        
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(image_data)
        mc.ComputeNormalsOn()
        contour_val = 0.5 if mask_np.dtype == np.uint8 or mask_np.dtype == bool else 0.3
        mc.SetValue(0, contour_val)
        mc.Update()
        poly_data = mc.GetOutput()
        
        if smoothing:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(poly_data)
            smoother.SetNumberOfIterations(30)
            smoother.BoundarySmoothingOn()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(0.1)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            poly_data = smoother.GetOutput()
            
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetAmbient(0.1)
        actor.GetProperty().SetDiffuse(0.8)
        return actor

    def update_volume_data(self):
        """Rebuilds the anatomical and calcium surfaces."""
        if hasattr(self, 'scene_3d_actors'):
            for actor in self.scene_3d_actors:
                if actor: self.ren_3d.RemoveActor(actor)
        self.scene_3d_actors = []
        
        heart_opacity = self.slider_tissue.value() / 100.0
        self.heart_actor = self._create_surface_actor(self.state.heart_mask, color=(0.9, 0.5, 0.5), opacity=heart_opacity)
        if self.heart_actor:
            self.ren_3d.AddActor(self.heart_actor)
            self.scene_3d_actors.append(self.heart_actor)
            
        mask = None
        if self.current_calc_mode == 'ai':
            mask = self.state.calc_mask
            vmask = self.state.vessel_mask
        elif self.current_calc_mode == 'gt':
            mask = self.state.gt_calc_mask
            vmask = None
            
        if mask is not None:
            if self.current_calc_mode == 'ai' and vmask is not None:
                colors = {1: (1, 1, 0), 2: (1, 0, 0), 3: (1, 0, 1), 4: (0, 0.5, 1)}
                for vid, color in colors.items():
                    v_mask_single = (vmask == vid).astype(np.uint8)
                    v_actor = self._create_surface_actor(v_mask_single, color=color, opacity=1.0)
                    if v_actor:
                        self.ren_3d.AddActor(v_actor)
                        self.scene_3d_actors.append(v_actor)
            else:
                c_actor = self._create_surface_actor(mask, color=(1, 0.9, 0.5), opacity=1.0)
                if c_actor:
                    self.ren_3d.AddActor(c_actor)
                    self.scene_3d_actors.append(c_actor)
                    
        self.ren_3d.ResetCamera()
        self.vtk_3d.GetRenderWindow().Render()
        self._update_mpr_overlays()

    def _update_mpr_overlays(self):
        if not hasattr(self, 'img_mask'): return
        
        mask = None
        if self.current_calc_mode == 'ai' and self.state.vessel_mask is not None:
            mask = self.state.vessel_mask
        elif self.current_calc_mode == 'gt' and self.state.gt_calc_mask is not None:
            mask = self.state.gt_calc_mask
        elif self.current_calc_mode == 'raw':
            mask = np.zeros_like(self.state.vol_hu, dtype=np.uint8)
            
        if mask is None:
            mask = np.zeros_like(self.state.vol_hu, dtype=np.uint8)
            
        vtk_data_mask = numpy_support.numpy_to_vtk(num_array=mask.astype(np.uint8).ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        self.img_mask.GetPointData().SetScalars(vtk_data_mask)
        self.img_mask.Modified()
        
        for vtk_w in [self.vtk_ax, self.vtk_cor, self.vtk_sag]:
            vtk_w.GetRenderWindow().Render()

    def update_transfer_functions(self):
        """Update heart opacity from slider."""
        if hasattr(self, 'heart_actor') and self.heart_actor:
            self.heart_actor.GetProperty().SetOpacity(self.slider_tissue.value() / 100.0)
            self.vtk_3d.GetRenderWindow().Render()

    def set_calcium_mode(self, mode):
        self.current_calc_mode = mode
        self.update_volume_data()
