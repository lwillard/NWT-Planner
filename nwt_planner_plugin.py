# -*- coding: utf-8 -*-
from qgis.PyQt.QtCore import Qt, QCoreApplication, QDateTime
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtWidgets import (
    QAction, QDialog, QVBoxLayout, QFormLayout,
    QLineEdit, QDialogButtonBox, QDateTimeEdit
)
from qgis.PyQt.QtCore import QSettings
from qgis.core import (
    QgsMessageLog, Qgis, QgsProject, QgsWkbTypes, QgsVectorLayer, QgsFeature, QgsField,
    QgsGeometry, QgsPointXY, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
    QgsTextFormat, QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsRendererCategory,
    QgsCategorizedSymbolRenderer, QgsFillSymbol, QgsMarkerSymbol, QgsUnitTypes
)
from qgis.core import (
    QgsProject, QgsField, QgsFeatureRequest, QgsTask, QgsApplication, Qgis
)

from qgis.core import QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat
from qgis.PyQt.QtGui import QFont, QColor

from qgis.gui import QgsMapTool, QgsRubberBand
from PyQt5.QtCore import QVariant
from pathlib import Path
import math
import numpy as np
from scipy import interpolate

from glasstone.overpressure import brode_overpressure

from .slide_rule_widget import SlideRuleWidget
import json, urllib.request, urllib.parse

# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate
from glasstone.overpressure import brode_overpressure

class OverpressureRingCalculator:
    def __init__(self, max_radius=25000, max_height=5000, radius_interval=100, height_interval=10):
        self.max_radius = max_radius
        self.max_height = max_height
        self.radius_interval = radius_interval
        self.height_interval = height_interval
        self.t = np.arange(0.01, self.max_radius, radius_interval)
        self.u = np.arange(0, self.max_height, height_interval)

    def GetOverpressureRadii(self, bomb_yield, height_of_burst, overpressures):
        ovps = brode_overpressure(bomb_yield, self.t, height_of_burst, 'kT', dunits='m', opunits='psi')
        f = interpolate.interp1d(ovps, self.t, bounds_error=False, fill_value=0.0)
        return f(overpressures)

def get_overpressure_radii(yield_kt: float, altitude_m: float):
    levels = [20, 5, 2, 1]
    calc = OverpressureRingCalculator()
    vals = calc.GetOverpressureRadii(yield_kt, altitude_m, levels)
    return {lvl: float(r) for lvl, r in zip(levels, vals)}



ELLIPSE_AXES_RATIO = 1.0
ELLIPSE_BEARING_DEG = 90.0
ELLIPSE_NUM_VERTICES = 180
PSI_ORDER_DESC = [20, 5, 2, 1]


class SlideRuleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NWT Planner")
        self.setModal(True)

        # The circular slide rule widget (visual rings/interaction)
        self.widget = SlideRuleWidget(self)

        # Text fields
        self.yield_edit = QLineEdit(self)
        self.height_edit = QLineEdit(self)
        self.target_edit = QLineEdit(self)

        self.yield_edit.setPlaceholderText("kT")
        self.height_edit.setPlaceholderText("meters")
        self.target_edit.setPlaceholderText("Target / label")

        # Restore last values
        s = QSettings()
        y = s.value("NWTPlanner/lastYield", 20, type=int)
        h = s.value("NWTPlanner/lastHeight", 0, type=int)
        t = s.value("NWTPlanner/lastTarget", "", type=str)
        dt = s.value("NWTPlanner/lastDateTimeISO", QDateTime.currentDateTime().toString(Qt.ISODate), type=str)

        self.widget.set_yield_value(int(y))
        self.widget.set_height_value(int(h))
        self.yield_edit.setText(str(int(y)))
        self.height_edit.setText(str(int(h)))
        self.target_edit.setText(t or "")

        # Date/time
        self.dt_edit = QDateTimeEdit(self)
        self.dt_edit.setCalendarPopup(True)
        try:
            self.dt_edit.setDateTime(QDateTime.fromString(dt, Qt.ISODate))
        except Exception:
            self.dt_edit.setDateTime(QDateTime.currentDateTime())

        # Wire up syncing
        self.widget.yieldChanged.connect(lambda v: self.yield_edit.setText(str(v)))
        self.widget.heightChanged.connect(lambda v: self.height_edit.setText(str(v)))
        self.yield_edit.editingFinished.connect(self._apply_yield_from_text)
        self.height_edit.editingFinished.connect(self._apply_height_from_text)

        # Layout
        form = QFormLayout()
        form.addRow("Yield (kT):", self.yield_edit)
        form.addRow("Burst height (m):", self.height_edit)
        form.addRow("Primary Target:", self.target_edit)
        form.addRow("Detonation time:", self.dt_edit)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        # dialog-level accept/reject; plugin will also hook into accepted to activate tool
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.addWidget(self.widget, 1)
        root.addLayout(form)
        root.addWidget(self.buttons)
        self.setLayout(root)

    def _apply_yield_from_text(self):
        try:
            self.widget.set_yield_value(int(self.yield_edit.text()))
        except Exception:
            pass

    def _apply_height_from_text(self):
        try:
            self.widget.set_height_value(int(self.height_edit.text()))
        except Exception:
            pass

    def values(self):
        # For compatibility with the requested activate_tool section
        try:
            y = float(self.yield_edit.text())
        except Exception:
            y = float(self.widget.get_yield_value())
        try:
            h = float(self.height_edit.text())
        except Exception:
            h = float(self.widget.get_height_value())
        t = self.target_edit.text().strip()
        return y, h, t

    def datetime_iso(self):
        return self.dt_edit.dateTime().toString(Qt.ISODate)

    def accept(self):
        # Persist dialog selections
        s = QSettings()
        s.setValue("NWTPlanner/lastYield", int(float(self.yield_edit.text() or "0")))
        s.setValue("NWTPlanner/lastHeight", int(float(self.height_edit.text() or "0")))
        s.setValue("NWTPlanner/lastTarget", self.target_edit.text())
        s.setValue("NWTPlanner/lastDateTimeISO", self.dt_edit.dateTime().toString(Qt.ISODate))
        super().accept()


class PreviewRingsTool(QgsMapTool):
    def __init__(self, plugin):
        super().__init__(plugin.canvas)
        self.plugin = plugin
        self.canvas = plugin.canvas
        self.setCursor(Qt.CrossCursor)
        self._bands = []
        self._last_point = None

    def deactivate(self):
        self._clear_bands()
        super().deactivate()

    def _clear_bands(self):
        for b in self._bands:
            try:
                b.reset(); b.hide(); b.deleteLater()
            except Exception:
                pass
        self._bands = []

    def _ensure_bands(self, count):
        while len(self._bands) < count:
            rb = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            rb.setStrokeColor(QColor(0, 0, 0, 180))
            rb.setWidth(1)
            rb.setFillColor(QColor(255, 80, 80, 60))
            self._bands.append(rb)
        while len(self._bands) > count:
            self._bands.pop().deleteLater()

    def canvasMoveEvent(self, e):
        p = self.toMapCoordinates(e.pos())
        self._last_point = p
        geoms = self.plugin._build_preview_geoms(p)
        self._ensure_bands(len(geoms))
        for band, (_, g_canvas) in zip(self._bands, geoms):
            band.setToGeometry(g_canvas, None)
            band.show()

    def canvasReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._last_point is not None:
            self.plugin._commit_at(self._last_point)
            self._clear_bands()
            self.plugin.iface.messageBar().pushSuccess("Overpressure Rings", "Placed rings.")
        elif e.button() == Qt.RightButton:
            self._clear_bands()
            self.plugin.iface.messageBar().pushInfo("Overpressure Rings", "Canceled.")


class NuclearWarTargetingPlannerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.action = None
        self.menu_text = "Vintage Slide Rule"
        self.icon_path = str(Path(__file__).parent / "icon.png")

        # Overpressure rings state
        self.tool = None
        self._ordered = []
        self._yield_kt = None
        self._altitude_m = None
        self._target_text = ""
        self._dt_iso = ""

    def tr(self, message):
        return QCoreApplication.translate("OverpressureRings", message)

    def initGui(self):
        try:
            self.action = QAction(QIcon(self.icon_path), self.menu_text, self.iface.mainWindow())
            self.action.triggered.connect(lambda checked=False: self.show_dialog())
            self.iface.addPluginToMenu(self.menu_text, self.action)
            self.iface.addToolBarIcon(self.action)
            QgsMessageLog.logMessage("VintageSlideRule loaded (dialog mode + overpressure rings).", "VintageSlideRule", Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"initGui error: {e}", "VintageSlideRule", Qgis.Critical)

    def show_dialog(self):
        try:
            dlg = SlideRuleDialog(self.iface.mainWindow())
            dlg.setWindowIcon(QIcon(self.icon_path))
            dlg.resize(560, 600)
            # When OK is clicked, activate the preview tool using the dialog's values
            dlg.buttons.accepted.connect(lambda: self._on_dialog_ok(dlg))
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as e:
            QgsMessageLog.logMessage(f"show_dialog error: {e}", "VintageSlideRule", Qgis.Critical)

    def _on_dialog_ok(self, dlg: SlideRuleDialog):
        try:
            # As requested: moved activate_tool logic runs on OK
            self._yield_kt, self._altitude_m, self._target_text = dlg.values()
            self._dt_iso = dlg.datetime_iso()  # new: capture detonation time

            radii = get_overpressure_radii(self._yield_kt, self._altitude_m)
            ordered = [(20, radii.get(20)), (5, radii.get(5)), (2, radii.get(2)), (1, radii.get(1))]
            self._ordered = [(psi, r) for psi, r in ordered if r and r > 0]

            self.tool = PreviewRingsTool(self)
            self.iface.mapCanvas().setMapTool(self.tool)
            self.iface.messageBar().pushInfo(self.tr("Overpressure Rings"),
                                             self.tr("Move mouse to preview. Left-click to place, right-click to cancel."))
        except Exception as e:
            QgsMessageLog.logMessage(f"_on_dialog_ok error: {e}", "VintageSlideRule", Qgis.Critical)

    def unload(self):
        try:
            if self.action is not None:
                self.iface.removeToolBarIcon(self.action)
                self.iface.removePluginMenu(self.menu_text, self.action)
                self.action.deleteLater()
                self.action = None
            QgsMessageLog.logMessage("VintageSlideRule unloaded.", "VintageSlideRule", Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"unload error: {e}", "VintageSlideRule", Qgis.Critical)

    # --------------- Ported helpers from OverpressureRingsPlugin ---------------
    def _build_preview_geoms(self, p_canvas_xy):
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        web_merc = QgsCoordinateReferenceSystem.fromEpsgId(3857)
        geoms = []
        for psi, r_m in self._ordered:
            g_3857 = self._ellipse_polygon_3857(p_canvas_xy, r_m, canvas_crs)
            g_canvas = self._transform_geom(g_3857, web_merc, canvas_crs)
            geoms.append((psi, g_canvas))
        return geoms

    def _commit_at(self, p_canvas_xy):
        poly_layer = self._get_or_create_polygons_layer()
        point_layer = self._get_or_create_points_layer(poly_layer)
        if poly_layer is None or point_layer is None:
            return

        if poly_layer.fields().indexOf("id") == -1:
            poly_layer.startEditing()
            poly_layer.addAttribute(QgsField("id", QVariant.Int))
            poly_layer.commitChanges()
        next_id = self._next_id(poly_layer)

        # Add new rings (raw)
        poly_layer.startEditing()
        try:
            for psi, r_m in self._ordered:
                g_layer = self._ellipse_geometry_at_point(p_canvas_xy, r_m, poly_layer)
                g_layer = self._clean(g_layer)
                f = QgsFeature(poly_layer.fields()); f.setGeometry(g_layer)
                f["id"] = next_id; next_id += 1
                f["psi"] = psi; f["radius_m"] = float(r_m)
                f["yield_kt"] = float(self._yield_kt); f["altitude_m"] = float(self._altitude_m)
                f["target"] = str(self._target_text)
                poly_layer.addFeature(f)
        finally:
            poly_layer.commitChanges(); poly_layer.triggerRepaint()

        # Nearest TARGET + center point
        info = self._nearest_target_info(p_canvas_xy)
        self._add_center_point(point_layer, p_canvas_xy, info)

        # Rebuild donuts and apply style
        self._rebuild_by_global_donut(poly_layer)
        self._ensure_labeling(poly_layer)
        self._apply_style(poly_layer)

    # ---------- Nearest target logic ----------
    def _nearest_target_info(self, p_canvas_xy):
        project = QgsProject.instance(); ctx = project.transformContext()
        canvas_crs = self.canvas.mapSettings().destinationCrs()
        to_wgs84 = QgsCoordinateTransform(canvas_crs, QgsCoordinateReferenceSystem.fromEpsgId(4326), ctx)
        p_ll = to_wgs84.transform(p_canvas_xy)
        best = {
            "LAT": float(p_ll.y()), "LON": float(p_ll.x()),
            "NAME": self._target_text or "",
            "H": "", "ST": "", "CLASS": "", "SUBCLASS": "", "NUM_1": "", "UNIT_1": ""
        }

        target_layer = None
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.name() == "TARGET_LIST":
                target_layer = lyr; break
        if not target_layer:
            return best

        tl_crs = target_layer.crs()
        to_canvas_from_tl = QgsCoordinateTransform(tl_crs, canvas_crs, ctx)
        to_canvas_from_wgs = QgsCoordinateTransform(QgsCoordinateReferenceSystem.fromEpsgId(4326), canvas_crs, ctx)

        has_lat = target_layer.fields().indexOf("LAT") != -1
        has_lon = target_layer.fields().indexOf("LON") != -1

        min_d2 = float("inf")
        nearest = None

        for f in target_layer.getFeatures():
            try:
                if f.geometry() and not f.geometry().isEmpty():
                    geom = f.geometry()
                    pt = geom.asPoint() if geom.wkbType() == QgsWkbTypes.PointGeometry else geom.centroid().asPoint()
                    p_can = to_canvas_from_tl.transform(pt)
                elif has_lat and has_lon and f["LAT"] is not None and f["LON"] is not None:
                    lat = float(f["LAT"]); lon = float(f["LON"])
                    p_can = to_canvas_from_wgs.transform(QgsPointXY(lon, lat))
                else:
                    continue
                dx = p_can.x() - p_canvas_xy.x(); dy = p_can.y() - p_canvas_xy.y()
                d2 = dx*dx + dy*dy
                if d2 < min_d2:
                    min_d2 = d2; nearest = f

            except Exception as e:
                self.show_exception_to_user(e)
                continue

        if nearest is None:
            self.iface.messageBar().pushMessage("Error", "No nearest point found", level=Qgis.Critical, duration=5)
            return best

        def get_field(ftr, name, default=""):
            idx = ftr.fields().indexOf(name)
            if idx == -1:
                return default
            val = ftr[name]
            return val if val is not None else default

        # Keep lat/lon from user's click in WGS84
        best["LAT"] = float(p_ll.y()); best["LON"] = float(p_ll.x())

        for k in ["NAME","H","ST","CLASS","SUBCLASS","NUM_1","UNIT_1"]:
            best[k] = get_field(nearest, k, best.get(k, ""))
        return best

    def show_exception_to_user(self, exception):
        self.iface.messageBar().pushMessage("Error", str(exception), level=Qgis.Critical, duration=5)

    def _add_center_point(self, point_layer, p_canvas_xy, info: dict):
        project = QgsProject.instance(); ctx = project.transformContext()
        canvas_crs = self.canvas.mapSettings().destinationCrs(); layer_crs = point_layer.crs()
        to_layer = QgsCoordinateTransform(canvas_crs, layer_crs, ctx)
        p_layer = to_layer.transform(p_canvas_xy)

        # ensure all fields exist
        needed = ["LAT","LON","yield_kt","altitude_m","NAME","H","ST","CLASS","SUBCLASS","NUM_1","UNIT_1","DT_ISO",
                "CITY","STATE","COUNTRY","NEAR_FEATURE"]
        point_layer.startEditing()
        try:
            for fld in needed:
                if point_layer.fields().indexOf(fld) == -1:
                    t = QVariant.Double if fld in ("LAT","LON","yield_kt","altitude_m") else QVariant.String
                    point_layer.addAttribute(QgsField(fld, t))
            point_layer.updateFields()

            # Lat/Lon in WGS84 already available from 'info' (your earlier code set this)
            lat = float(info.get("LAT", 0.0))
            lon = float(info.get("LON", 0.0))

            # Online lookups (best-effort; empty strings on failure)
            ge = self._reverse_geocode_osm(lat, lon)          # {"CITY","STATE","COUNTRY"}
            near_name = self._nearest_osm_feature_name(lat, lon)  # "Some Park" etc.

            f = QgsFeature(point_layer.fields()); f.setGeometry(QgsGeometry.fromPointXY(p_layer))
            f["LAT"] = lat; f["LON"] = lon
            f["NAME"] = info.get("NAME",""); f["H"] = info.get("H",""); f["ST"] = info.get("ST","")
            f["CLASS"] = info.get("CLASS",""); f["SUBCLASS"] = info.get("SUBCLASS","")
            f["NUM_1"] = info.get("NUM_1",""); f["UNIT_1"] = info.get("UNIT_1","")
            f["yield_kt"] = float(self._yield_kt or 0.0); f["altitude_m"] = float(self._altitude_m or 0.0)
            f["DT_ISO"] = str(self._dt_iso or QDateTime.currentDateTime().toString(Qt.ISODate))
            # NEW:
            f["CITY"] = ge.get("CITY",""); f["STATE"] = ge.get("STATE",""); f["COUNTRY"] = ge.get("COUNTRY","")
            f["NEAR_FEATURE"] = near_name

            point_layer.addFeature(f)
        finally:
            point_layer.commitChanges(); point_layer.triggerRepaint()

    def _next_id(self, layer):
        try:
            idx = layer.fields().indexOf("id")
            mv = layer.maximumValue(idx)
            return (int(mv) if mv is not None else 0) + 1
        except Exception:
            mx = 0
            for f in layer.getFeatures():
                try:
                    mx = max(mx, int(f["id"]))
                except Exception:
                    pass
            return mx + 1

    def _rebuild_by_global_donut(self, layer):
        bypsi = {}
        for f in layer.getFeatures():
            try:
                psi = int(f["psi"])
            except Exception:
                continue
            bypsi.setdefault(psi, []).append(f)

        unions = {}
        for psi, feats in bypsi.items():
            geoms = [self._clean(f.geometry()) for f in feats if f.geometry()]  # type: ignore
            if not geoms:
                continue
            unions[psi] = self._clean(QgsGeometry.unaryUnion(geoms))

        higher_union = QgsGeometry()
        layer.startEditing()
        try:
            to_delete = [f.id() for f in layer.getFeatures() if f.fields().indexOf("psi") != -1]
            if to_delete:
                layer.deleteFeatures(to_delete)

            for psi in PSI_ORDER_DESC:
                U = unions.get(psi)
                if not U or U.isEmpty():
                    continue
                mask = self._clean(U.difference(higher_union)) if (higher_union and not higher_union.isEmpty()) else U
                # explode into parts
                parts = self._explode(mask)
                for part in parts:
                    nf = QgsFeature(layer.fields()); nf.setGeometry(part)
                    nf["id"] = self._next_id(layer)
                    nf["psi"] = psi
                    layer.addFeature(nf)
                higher_union = U if (not higher_union or higher_union.isEmpty()) else self._clean(QgsGeometry.unaryUnion([higher_union, U]))
        finally:
            layer.commitChanges(); layer.triggerRepaint()

    def _apply_style(self, layer):
        stroke_mm = 0.2 * 0.352777778
        presets = {
            20: (QColor(255, 0, 0, 153), QColor(0, 0, 0, 255)),
            5:  (QColor(255, 140, 0, 102), QColor(0, 0, 0, 255)),
            2:  (QColor(255, 165, 0, 77),  QColor(0, 0, 0, 255)),
            1:  (QColor(255, 255, 0, 51),  QColor(0, 0, 0, 255)),
        }
        cats = []
        for psi, (fill, outline) in presets.items():
            sym = QgsFillSymbol.createSimple({})
            sym.setColor(fill)
            sl = sym.symbolLayer(0)
            sl.setStrokeColor(outline)
            sl.setStrokeWidth(stroke_mm)
            sl.setStrokeWidthUnit(QgsUnitTypes.RenderMillimeters)
            cats.append(QgsRendererCategory(psi, sym, f"{psi} psi"))
        renderer = QgsCategorizedSymbolRenderer("psi", cats)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

    def _ensure_labeling(self, layer: QgsVectorLayer):
        # Define the label expression
        expr = 'to_string("psi") || \' psi\''
        
        # Create PAL labeling settings
        s = QgsPalLayerSettings()
        s.fieldName = expr
        s.isExpression = True
        
        # Set placement to Perimeter (use PerimeterCurved for QGIS 3.34)
        s.placement = QgsPalLayerSettings.PerimeterCurved  # Curved text along polygon boundaries
        
        # Optional: Distance from the perimeter
        s.dist = 1.0  # Distance in map units (adjust as needed)
        s.priority = 10  # Higher priority for label placement (1-10)
        s.labelAllParts = True  # Label all parts of multipart polygons
        s.repeatDistance = 0  # Prevent repeating labels along perimeter
        
        # Configure text format
        fmt = QgsTextFormat()
        fmt.setFont(QFont("Arial", 10))
        fmt.setSize(10)
        fmt.setColor(QColor(0, 0, 0))  # Black text
        s.setFormat(fmt)
        
        # Apply labeling to the layer
        layer.setLabeling(QgsVectorLayerSimpleLabeling(s))
        layer.setLabelsEnabled(True)
        layer.triggerRepaint()

    # ---- Geometry helpers ----
    def _ellipse_polygon_3857(self, point_canvas_xy, radius_m, canvas_crs) -> QgsGeometry:
        project = QgsProject.instance(); ctx = project.transformContext()
        web_merc = QgsCoordinateReferenceSystem.fromEpsgId(3857)
        to_3857 = QgsCoordinateTransform(canvas_crs, web_merc, ctx)
        center_3857 = to_3857.transform(point_canvas_xy)
        a = radius_m * max(ELLIPSE_AXES_RATIO, 1e-9)
        b = radius_m / max(ELLIPSE_AXES_RATIO, 1e-9)
        theta = (ELLIPSE_BEARING_DEG or 0.0) * math.pi / 180.0
        pts = []
        for i in range(ELLIPSE_NUM_VERTICES):
            t = 2 * math.pi * i / ELLIPSE_NUM_VERTICES
            x = a * math.cos(t); y = b * math.sin(t)
            xr = x * math.cos(theta) - y * math.sin(theta)
            yr = x * math.sin(theta) + y * math.cos(theta)
            pts.append(QgsPointXY(center_3857.x() + xr, center_3857.y() + yr))
        pts.append(pts[0])
        return QgsGeometry.fromPolygonXY([pts])

    def _transform_geom(self, geom, src_crs, dst_crs):
        g = QgsGeometry(geom)
        g.transform(QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance().transformContext()))
        return g

    def _ellipse_geometry_at_point(self, point_canvas_xy, radius_m, layer):
        project = QgsProject.instance(); context = project.transformContext()
        canvas_crs = self.canvas.mapSettings().destinationCrs(); layer_crs = layer.crs()
        web_merc = QgsCoordinateReferenceSystem.fromEpsgId(3857)
        to_3857 = QgsCoordinateTransform(canvas_crs, web_merc, context)
        to_layer = QgsCoordinateTransform(web_merc, layer_crs, context)
        center_3857 = to_3857.transform(point_canvas_xy)
        a = radius_m * max(ELLIPSE_AXES_RATIO, 1e-9); b = radius_m / max(ELLIPSE_AXES_RATIO, 1e-9)
        theta = (ELLIPSE_BEARING_DEG or 0.0) * math.pi / 180.0
        pts = []
        for i in range(ELLIPSE_NUM_VERTICES):
            t = 2 * math.pi * i / ELLIPSE_NUM_VERTICES
            x = a * math.cos(t); y = b * math.sin(t)
            xr = x * math.cos(theta) - y * math.sin(theta)
            yr = x * math.sin(theta) + y * math.cos(theta)
            pts.append(QgsPointXY(center_3857.x() + xr, center_3857.y() + yr))
        pts.append(pts[0])
        poly_3857 = QgsGeometry.fromPolygonXY([pts])
        poly_layer = QgsGeometry(poly_3857); poly_layer.transform(to_layer)
        return poly_layer

    # ---- Layers ----
    def _get_or_create_polygons_layer(self):
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.name() == "PSI_Rings" and lyr.geometryType() == QgsWkbTypes.PolygonGeometry:
                needed = ["id","psi","radius_m","yield_kt","altitude_m","target"]
                if any(lyr.fields().indexOf(n)==-1 for n in needed):
                    lyr.startEditing()
                    for n in needed:
                        if lyr.fields().indexOf(n)==-1:
                            t = QVariant.Int if n in ("id","psi") else (QVariant.Double if n in ("radius_m","yield_kt","altitude_m") else QVariant.String)
                            lyr.addAttribute(QgsField(n, t))
                    lyr.commitChanges()
                return lyr

        layer = self.iface.activeLayer()
        if isinstance(layer, QgsVectorLayer) and layer.geometryType() == QgsWkbTypes.PolygonGeometry:
            needed = ["id","psi","radius_m","yield_kt","altitude_m","target"]
            if any(layer.fields().indexOf(n)==-1 for n in needed):
                layer.startEditing()
                for n in needed:
                    if layer.fields().indexOf(n)==-1:
                        t = QVariant.Int if n in ("id","psi") else (QVariant.Double if n in ("radius_m","yield_kt","altitude_m") else QVariant.String)
                        layer.addAttribute(QgsField(n, t))
                layer.commitChanges()
            if layer.name() != "PSI_Rings":
                layer.setName("PSI_Rings")
            return layer

        vl = QgsVectorLayer("Polygon?crs=EPSG:3857", "PSI_Rings", "memory")
        pr = vl.dataProvider()
        pr.addAttributes([QgsField("id", QVariant.Int), QgsField("psi", QVariant.Int),
                            QgsField("radius_m", QVariant.Double), QgsField("yield_kt", QVariant.Double),
                            QgsField("altitude_m", QVariant.Double), QgsField("target", QVariant.String)])
        vl.updateFields(); QgsProject.instance().addMapLayer(vl)
        return vl

    def _get_or_create_points_layer(self, poly_layer):
        name = "PSI_Centers"
        for lyr in QgsProject.instance().mapLayers().values():
            if isinstance(lyr, QgsVectorLayer) and lyr.name() == name and lyr.geometryType() == QgsWkbTypes.PointGeometry:
                return lyr
        crs = poly_layer.crs().authid() if poly_layer else self.canvas.mapSettings().destinationCrs().authid()
        vl = QgsVectorLayer(f"Point?crs={crs}", name, "memory")
        pr = vl.dataProvider()
        pr.addAttributes([
            QgsField("LAT", QVariant.Double), QgsField("LON", QVariant.Double),
            QgsField("yield_kt", QVariant.Double), QgsField("altitude_m", QVariant.Double),
            QgsField("NAME", QVariant.String), QgsField("H", QVariant.String),
            QgsField("ST", QVariant.String), QgsField("CLASS", QVariant.String),
            QgsField("SUBCLASS", QVariant.String), QgsField("NUM_1", QVariant.String),
            QgsField("UNIT_1", QVariant.String), QgsField("DT_ISO", QVariant.String),
            # NEW:
            QgsField("CITY", QVariant.String), QgsField("STATE", QVariant.String),
            QgsField("COUNTRY", QVariant.String), QgsField("NEAR_FEATURE", QVariant.String),
            QgsField("LU_KEY", QVariant.String)
        ])
        vl.updateFields(); QgsProject.instance().addMapLayer(vl)
        sym = QgsMarkerSymbol.createSimple({"name":"circle","size":"4"}); vl.renderer().setSymbol(sym)
        return vl

    # ---- Utility geometry ops ----
    def _clean(self, g: QgsGeometry) -> QgsGeometry:
        if not g or g.isEmpty():
            return QgsGeometry()
        try:
            g = g.makeValid()
        except Exception:
            pass
        try:
            g = g.buffer(0, 8)
        except Exception:
            pass
        return g

    def _explode(self, g: QgsGeometry):
        if not g or g.isEmpty():
            return []
        try:
            if g.isMultipart():
                return [QgsGeometry.fromPolygonXY(p) for p in g.asMultiPolygon()]
            return [g]
        except Exception:
            return [g]

    def _nearest_osm_feature_name(self, lat: float, lon: float, radius_m: int = 1200) -> str:
        """
        Returns nearest 'name' from OSM (node/way/relation) within radius_m for selected
        non-road feature categories (military, port/harbour, refinery/industrial works,
        mall, townhall/government, dams, power plants/substations, universities/colleges).
        Roads/highways are excluded by construction.
        """
        url = "https://overpass-api.de/api/interpreter"
        headers = {
            "User-Agent": "VintageSlideRule/1.0 (QGIS plugin)",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        # ——— Filtered feature sets (each requires a name=*) ———
        # amenity-based civic/education:
        #   universities/colleges, town/city hall (government seat)
        # office:
        #   government offices
        # commerce:
        #   malls
        # power:
        #   power plants & substations
        # water infra:
        #   dams (both man_made=dam and waterway=dam used in the wild)
        # industrial:
        #   industrial=refinery / chemical / factory, man_made=works (general works),
        #   landuse=industrial (named areas), landuse=harbour & harbour=*
        # military:
        #   military=* or landuse=military
        q = f"""
    [out:json][timeout:20];
    (
    // Universities / colleges / city halls
    node(around:{radius_m},{lat},{lon})["name"]["amenity"~"^(university|college|townhall)$"];
    way(around:{radius_m},{lat},{lon})["name"]["amenity"~"^(university|college|townhall)$"];
    relation(around:{radius_m},{lat},{lon})["name"]["amenity"~"^(university|college|townhall)$"];

    // Government offices
    node(around:{radius_m},{lat},{lon})["name"]["office"="government"];
    way(around:{radius_m},{lat},{lon})["name"]["office"="government"];
    relation(around:{radius_m},{lat},{lon})["name"]["office"="government"];

    // Malls
    node(around:{radius_m},{lat},{lon})["name"]["shop"="mall"];
    way(around:{radius_m},{lat},{lon})["name"]["shop"="mall"];
    relation(around:{radius_m},{lat},{lon})["name"]["shop"="mall"];

    // Power plants / substations
    node(around:{radius_m},{lat},{lon})["name"]["power"~"^(plant|substation)$"];
    way(around:{radius_m},{lat},{lon})["name"]["power"~"^(plant|substation)$"];
    relation(around:{radius_m},{lat},{lon})["name"]["power"~"^(plant|substation)$"];

    // Dams
    node(around:{radius_m},{lat},{lon})["name"]["man_made"="dam"];
    way(around:{radius_m},{lat},{lon})["name"]["man_made"="dam"];
    relation(around:{radius_m},{lat},{lon})["name"]["man_made"="dam"];
    node(around:{radius_m},{lat},{lon})["name"]["waterway"="dam"];
    way(around:{radius_m},{lat},{lon})["name"]["waterway"="dam"];
    relation(around:{radius_m},{lat},{lon})["name"]["waterway"="dam"];

    // Refineries / industrial works
    node(around:{radius_m},{lat},{lon})["name"]["industrial"~"^(refinery|chemical|factory)$"];
    way(around:{radius_m},{lat},{lon})["name"]["industrial"~"^(refinery|chemical|factory)$"];
    relation(around:{radius_m},{lat},{lon})["name"]["industrial"~"^(refinery|chemical|factory)$"];
    node(around:{radius_m},{lat},{lon})["name"]["man_made"="works"];
    way(around:{radius_m},{lat},{lon})["name"]["man_made"="works"];
    relation(around:{radius_m},{lat},{lon})["name"]["man_made"="works"];
    node(around:{radius_m},{lat},{lon})["name"]["landuse"="industrial"];
    way(around:{radius_m},{lat},{lon})["name"]["landuse"="industrial"];
    relation(around:{radius_m},{lat},{lon})["name"]["landuse"="industrial"];

    // Ports / harbours
    node(around:{radius_m},{lat},{lon})["name"]["landuse"="harbour"];
    way(around:{radius_m},{lat},{lon})["name"]["landuse"="harbour"];
    relation(around:{radius_m},{lat},{lon})["name"]["landuse"="harbour"];
    node(around:{radius_m},{lat},{lon})["name"]["harbour"~"^(yes|port)$"];
    way(around:{radius_m},{lat},{lon})["name"]["harbour"~"^(yes|port)$"];
    relation(around:{radius_m},{lat},{lon})["name"]["harbour"~"^(yes|port)$"];

    // Military bases / areas
    node(around:{radius_m},{lat},{lon})["name"]["military"];
    way(around:{radius_m},{lat},{lon})["name"]["military"];
    relation(around:{radius_m},{lat},{lon})["name"]["military"];
    node(around:{radius_m},{lat},{lon})["name"]["landuse"="military"];
    way(around:{radius_m},{lat},{lon})["name"]["landuse"="military"];
    relation(around:{radius_m},{lat},{lon})["name"]["landuse"="military"];
    );
    out tags center 60;
    """
        data = urllib.parse.urlencode({"data": q}).encode("utf-8")

        try:
            js = self._http_get_json(url, headers=headers, data=data, timeout=25)
            els = js.get("elements", []) if isinstance(js, dict) else []
            if not els:
                return ""

            # pick the closest by distance (haversine) based on node coords or way/relation center
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000.0
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
                return 2 * R * math.asin(math.sqrt(a))

            best = None; best_d = 1e20
            for e in els:
                name = e.get("tags", {}).get("name")
                if not name:
                    continue
                if "lat" in e and "lon" in e:
                    clat, clon = float(e["lat"]), float(e["lon"])
                elif "center" in e and isinstance(e["center"], dict):
                    clat, clon = float(e["center"]["lat"]), float(e["center"]["lon"])
                else:
                    continue
                d = haversine(lat, lon, clat, clon)
                if d < best_d:
                    best_d, best = d, name
            return best or ""
        except Exception as e:
            QgsMessageLog.logMessage(f"Overpass nearest feature failed: {e}", "VintageSlideRule", Qgis.Warning)
            return ""

    def _http_get_json(self, url, headers=None, data=None, timeout=10):
        req = urllib.request.Request(url, data=data, headers=headers or {}, method="GET" if data is None else "POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _reverse_geocode_osm(self, lat: float, lon: float) -> dict:
        """
        Returns {'CITY':..., 'STATE':..., 'COUNTRY':...} using OSM Nominatim.
        """
        q = {
            "format": "jsonv2",
            "lat": f"{lat:.8f}",
            "lon": f"{lon:.8f}",
            "zoom": "12",
            "addressdetails": "1"
        }
        q["email"] = "castlebravo92@gmail.com"
        url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(q)
        headers = {
            "User-Agent": "VintageSlideRule/1.0 (QGIS plugin)",
            "Accept": "application/json"
        }
        try:
            js = self._http_get_json(url, headers=headers, timeout=10)
            addr = js.get("address", {}) if isinstance(js, dict) else {}
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or ""
            state = addr.get("state") or addr.get("region") or addr.get("province") or ""
            country = addr.get("country") or ""
            return {"CITY": city, "STATE": state, "COUNTRY": country}
        except Exception as e:
            QgsMessageLog.logMessage(f"Reverse geocode failed: {e}", "VintageSlideRule", Qgis.Warning)
            return {"CITY": "", "STATE": "", "COUNTRY": ""}
