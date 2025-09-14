# -*- coding: utf-8 -*-
import math
from qgis.PyQt.QtCore import Qt, QPoint, QRectF, pyqtSignal, QDateTime, QSettings
from qgis.PyQt.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath
from qgis.PyQt.QtWidgets import QWidget, QDateTimeEdit
from qgis.core import QgsMessageLog, Qgis
from qgis.PyQt.QtGui import QFontMetricsF
from glasstone.overpressure import brode_overpressure

# used for interpolation for nuclear blast effects
from scipy import interpolate

# used for interpolation for nuclear blast effects
import numpy as np

_TICK_LABEL_OFFSET_PX = 8

class YieldAltitudeOptimizer:
    def __init__(self, max_radius=25000, max_height=5000, radius_interval=100, height_interval=10):
        self.max_radius = max_radius
        self.max_height = max_height
        self.radius_interval = radius_interval
        self.height_interval = height_interval

        # distance/radius intervals
        self.t = np.arange(0.01, self.max_radius, radius_interval)

        # height of detonation intervals
        self.u = np.arange(0, self.max_height, height_interval)

    # for a given bomb yield and desired over-pressure, calculates and returns the optimal detonation height to maximize the radius of the given overpressure
    def optimize_for_overpressure(self, bomb_yield, overpressure):
        arrs = []

        for height_of_burst in self.u:
            ovps = brode_overpressure(bomb_yield, self.t, height_of_burst, 'kT', dunits='m', opunits='psi')   

            # this defines a function that calculates/interpolates the extent of the radius where an overpressure is found for a given yield and height of burst 
            f = interpolate.interp1d(ovps, self.t, bounds_error=False, fill_value=0.0)

            # append to the array
            arrs.append(f(overpressure))

        # find the max radius of the overpressure across all of the heights
        max_radius = max(arrs)

        # find the index of the max radius
        max_index = arrs.index(max_radius)

        # return the value for the index of the height array used to generate the max radius
        return self.u[max_index]

class SlideRuleWidget(QWidget):
    yieldChanged = pyqtSignal(int)
    heightChanged = pyqtSignal(int)
    dateTimeChanged = pyqtSignal(QDateTime)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Ranges
        self.yield_min, self.yield_max = 1, 100000
        self.height_min_positive, self.height_max = 1, 100000

        self._opt_hob_cache = {}  # key: (psi:int, yield_kt:float) -> altitude_m:float
        self._yao = YieldAltitudeOptimizer(
            max_radius=25000, max_height=5000, radius_interval=100, height_interval=10
        )

        # State
        self._yield_val = 100
        self._height_val = 100
        # Geometry
        self.start_angle_deg = -210
        self.sweep_deg = 300
        self.pad = 12
        self.outer_radius_ratio = 0.90
        self.yield_ring_ratio = 0.80
        self.height_ring_ratio = 0.60
        self.ring_thickness = 16
        self.height_band_thickness = int(self.ring_thickness * 1.9)
        self.indicator_length = max(20, int(self.ring_thickness * 1.2))

        # Colors
        self.col_bg = QColor('#ffffff')
        self.col_ring = QColor('#333333')
        self.col_minor = QColor('#888888')
        self.col_text = QColor('#222222')
        self.col_yield = QColor(0, 122, 204)
        self.col_height = QColor(200, 80, 0)
        self.col_zero_label = QColor(160, 0, 0)
        # Bands visual
        self.band_colors = [
            QColor(220, 0, 0, 110),
            QColor(255, 140, 0, 110),
            QColor(255, 215, 0, 110),
            QColor(0, 160, 0, 110)
        ]
        self.band_labels = ['ground', 'low air', 'air', 'high air']
        # Legacy fractional params (fallback)
        self._band_base = [0.10, 0.35, 0.70]
        self._band_gain = [0.15, 0.15, 0.10]
        # Meter-driven params via QSettings
        s = QSettings()
        self._band_mode = s.value('CircularSlideRule/band_mode', None)
        self._bounds_min = self._csv3('CircularSlideRule/band_bounds_min', [500.0, 2000.0, 8000.0])
        self._bounds_max = self._csv3('CircularSlideRule/band_bounds_max', [5000.0, 15000.0, 40000.0])
        self._widths_min = self._csv3('CircularSlideRule/band_widths_min', [500.0, 1500.0, 6000.0])
        self._widths_max = self._csv3('CircularSlideRule/band_widths_max', [3000.0, 10000.0, 30000.0])
        # Interaction
        self._active_ring = None
        self.setMouseTracking(True)
        # Center DateTime (kept inside inner core)
        self.dt_edit = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.dt_edit.setCalendarPopup(True)
        self.dt_edit.setDisplayFormat('yyyy-MM-dd HH:mm:ss')
        self.dt_edit.dateTimeChanged.connect(self.dateTimeChanged)
        self.dt_edit.setObjectName('centerDateTime')
        self.dt_edit.setStyleSheet('QDateTimeEdit { background: rgba(255,255,255,235); }')

    def _draw_text_polar(self, painter, cx, cy, radius_px, angle_deg, text,
                        center=True, rotate=False, radial_offset_px=_TICK_LABEL_OFFSET_PX):
        """
        Draw `text` at a point on a circle centered at (cx,cy), at `radius_px` and `angle_deg`.
        - If center=True, the text is centered on that point.
        - If rotate=True, the text is rotated so its baseline is tangent to the circle.
        """
        theta = math.radians(angle_deg)
        # point at tick end
        x = cx + radius_px * math.cos(theta)
        y = cy - radius_px * math.sin(theta)  # NOTE: Y axis down in Qt

        # nudge label outward along the radius (away from the circle)
        x_out = x + radial_offset_px * math.cos(theta)
        y_out = y - radial_offset_px * math.sin(theta)

        fm = QFontMetricsF(painter.font())
        w = fm.horizontalAdvance(text)
        h = fm.height()

        painter.save()
        if rotate:
            # rotate so text is tangent but still readable
            painter.translate(x_out, y_out)
            painter.rotate(-angle_deg)               # baseline along tangent
            rect = QRectF(-w/2.0, -h/2.0, w, h) if center else QRectF(0, -h, w, h)
            painter.drawText(rect, Qt.AlignCenter if center else Qt.AlignLeft | Qt.AlignVCenter, text)
        else:
            # keep upright; just center the rect on the point
            rect = QRectF(x_out - w/2.0, y_out - h/2.0, w, h) if center else QRectF(x_out, y_out - h, w, h)
            painter.drawText(rect, Qt.AlignCenter if center else Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.restore()        

    def _angle_for_value(self, val, vmin, vmax, ang_start_deg, ang_span_deg):
        # clockwise increasing; adjust sign if your dial is CCW
        t = (math.log10(val) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        return ang_start_deg + t * ang_span_deg        

    def _csv3(self, key, default):
        s = QSettings()
        try:
            parts = [float(x.strip()) for x in str(s.value(key, ','.join(str(v) for v in default))).split(',')]
            return parts if len(parts) == 3 else default
        except Exception:
            return default

    # ---- Utility ----
    def _clamp(self, v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    @staticmethod
    def get_fireball_diameter(yield_kt: float) -> float:
        return 75.0 * (yield_kt ** 0.4)

    @staticmethod
    def high_airburst_range(yield_kt: float) -> float:
        return 2230.0 * (yield_kt ** (1.0 / 3.0))

    # ---- Config API ----
    def set_band_parameters(self, base, gain):
        try:
            if len(base) == 3 and len(gain) == 3:
                self._band_base = [self._clamp(float(x), 0.0, 0.99) for x in base]
                self._band_gain = [self._clamp(float(x), 0.0, 1.0) for x in gain]
                self.update()
        except Exception as e:
            QgsMessageLog.logMessage(f'Invalid band params: {e}', 'NWT Planner', Qgis.Warning)

    def band_parameters(self):
        return list(self._band_base), list(self._band_gain)

    # ---- Date/time ----
    def set_date_time(self, dt: QDateTime):
        if isinstance(dt, QDateTime) and dt.isValid():
            if self.dt_edit.dateTime() != dt:
                self.dt_edit.blockSignals(True)
                self.dt_edit.setDateTime(dt)
                self.dt_edit.blockSignals(False)
                self.dateTimeChanged.emit(dt)

    def date_time(self):
        return self.dt_edit.dateTime()

    # ---- Public values ----
    def set_yield_value(self, v: int):
        v = int(self._clamp(v, self.yield_min, self.yield_max))
        if self._yield_val != v:
            self._yield_val = v
            self.yieldChanged.emit(v)
            self.update()

    def set_height_value(self, v: int):
        v = int(self._clamp(v, 0, self.height_max))
        if self._height_val != v:
            self._height_val = v
            self.heightChanged.emit(v)
            self.update()

    # ---- Painting ----
    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            rect = self.rect()
            size = min(rect.width(), rect.height())
            c = rect.center()
            radius = int(size * self.outer_radius_ratio / 2) - self.pad
            p.fillRect(rect, QBrush(self.col_bg))
            ry = int(radius * self.yield_ring_ratio)
            rh = int(radius * self.height_ring_ratio)
            self._draw_title(p)
            try:
                self._draw_height_bands(p, c, rh)
            except Exception as e:
                QgsMessageLog.logMessage(f'Band draw error: {e}', 'NWT Planner', Qgis.Warning)
            self._draw_log_ring(p, c, ry, 'Yield (kt)', self.yield_min, self.yield_max, include_zero=False)
            self._draw_log_ring(p, c, rh, 'Height (m)', self.height_min_positive, self.height_max, include_zero=True)
            self._draw_indicator(p, c, ry, self._yield_val, self.yield_min, self.yield_max, self.col_yield)
            h_val = self.height_min_positive if self._height_val == 0 else self._height_val
            self._draw_indicator(p, c, rh, h_val, self.height_min_positive, self.height_max, self.col_height, zero_mode=(self._height_val == 0))
            self._draw_legend(p)
            p.end()
        except Exception as e:
            QgsMessageLog.logMessage(f'paintEvent error: {e}', 'NWT Planner', Qgis.Critical)

    def resizeEvent(self, event):
        rect = self.rect()
        size = min(rect.width(), rect.height())
        c = rect.center()
        radius = int(size * self.outer_radius_ratio / 2) - self.pad
        rh = int(radius * self.height_ring_ratio)
        clear_r = max(56, rh - self.ring_thickness - 12)
        hint = self.dt_edit.sizeHint()
        max_w = max(120, int(clear_r * 2 - 8))
        w = min(max_w, max(140, hint.width()))
        h = max(22, hint.height())
        self.dt_edit.setGeometry(c.x() - w // 2, c.y() - h // 2, w, h)
        super().resizeEvent(event)

    def _draw_title(self, p):
        p.setPen(QPen(self.col_text, 1.5))
        f = p.font()
        f.setBold(True)
        f.setPointSize(10)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignTop | Qt.AlignHCenter, 'Dial-A-Yield')

    def _draw_legend(self, p):
        r = self.rect()
        f = p.font()
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QPen(self.col_yield, 2))
        p.drawText(r.adjusted(0, r.height() - 36, 0, -18), Qt.AlignHCenter | Qt.AlignVCenter, f'Yield (kt): {self._yield_val:,}')
        p.setPen(QPen(self.col_height, 2))
        p.drawText(r.adjusted(0, r.height() - 18, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter, f'Height (m): {self._height_val:,}')

    # ---- Bands (meter-driven or fractional fallback) ----
    def _compute_band_boundaries(self):
        y = self._clamp(self._yield_val, self.yield_min, self.yield_max)
        t = (math.log10(y) - math.log10(self.yield_min)) / (math.log10(self.yield_max) - math.log10(self.yield_min))
        H1 = self.get_fireball_diameter(float(y))  # ground top
        H2 = 2.0 * H1  # low-air top
        H3 = self.high_airburst_range(float(y))  # start of high-air
        # clamp and enforce monotonicity within 0 .. height_max
        H1 = self._clamp(H1, 1.0, self.height_max - 3.0)
        H2 = self._clamp(H2, H1 + 1.0, self.height_max - 2.0)
        H3 = self._clamp(H3, H2 + 1.0, self.height_max - 1.0)
        return [0, int(round(H1)), int(round(H2)), int(round(H3)), self.height_max]

    def _draw_height_bands(self, p, c, r):
        b = self._compute_band_boundaries()
        outer = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        inner_r = max(1, r - self.height_band_thickness)
        inner = QRectF(c.x() - inner_r, c.y() - inner_r, 2 * inner_r, 2 * inner_r)
        for i in range(4):
            v0, v1 = b[i], b[i + 1]
            a0 = self._value_to_angle_deg(max(self.height_min_positive, v0 if v0 > 0 else self.height_min_positive), self.height_min_positive, self.height_max)
            a1 = self._value_to_angle_deg(max(self.height_min_positive, v1 if v1 > 0 else self.height_min_positive), self.height_min_positive, self.height_max)
            start = max(self.start_angle_deg, min(self.start_angle_deg + self.sweep_deg, a0))
            end = max(self.start_angle_deg, min(self.start_angle_deg + self.sweep_deg, a1))
            if end <= start:
                continue
            path = QPainterPath()
            path.arcMoveTo(outer, -start)
            path.arcTo(outer, -start, -(end - start))
            path.arcTo(inner, -end, (end - start))
            path.closeSubpath()
            p.save()
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self.band_colors[i]))
            p.drawPath(path)
            p.restore()
            mid = (start + end) / 2.0
            self._draw_band_label(p, c, (inner_r + r) // 2, mid, self.band_labels[i])

    # ---- Selectors / Rings ----
    def _draw_band_label(self, p, c, rad, ang_deg, text):
        r = math.radians(ang_deg)
        s, cc = math.sin(r), math.cos(r)
        pt = QPoint(int(c.x() + rad * cc), int(c.y() + rad * s))
        p.save()
        p.setPen(QPen(QColor(0, 0, 0, 220), 1))
        f = p.font()
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(text)
        th = p.fontMetrics().height()
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)        
        p.drawText(pt.x() - tw // 2, pt.y() + th // 4, text)
        p.restore()

    def _draw_log_ring(self, p, c, r, label, vmin, vmax, include_zero=False):
        p.save()
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.setPen(QPen(self.col_ring, 3))
        p.setBrush(Qt.NoBrush)
        # drawArc uses 1/16 degrees; keep your original signs
        p.drawArc(rect, int(-self.start_angle_deg * 16), int(-self.sweep_deg * 16))

        dmin = int(math.floor(math.log10(max(vmin, 1e-9))))
        dmax = int(math.floor(math.log10(max(vmax, 1e-9))))

        major_len = self.ring_thickness * 0.95
        minor_len = self.ring_thickness * 0.55

        # regular decade + minor ticks (unchanged)
        for d in range(dmin, dmax + 1):
            base = 10 ** d
            if base < vmin or base > vmax:
                continue

            # Major tick at the decade
            self._draw_tick_at_value(p, c, r, base, vmin, vmax, major_len, QColor('#222222'))

            # Decade label
            ang = self._value_to_angle_deg(base, vmin, vmax)
            self._draw_text_on_ring(p, c, r, ang, self._fmt_k(base))

            # Minor ticks 2..9
            for m in range(2, 10):
                val = m * base
                if val < vmin or val > vmax:
                    continue
                self._draw_tick_at_value(p, c, r, val, vmin, vmax, minor_len, self.col_minor)

        # ==== OPTIMUM HOB TICKS (height ring only) ====
        if str(label).lower().startswith("height"):
            # Current yield (kT)
            try:
                yield_kt = float(self.get_yield_value() if hasattr(self, "get_yield_value") else self._yield_val)
            except Exception:
                yield_kt = 20.0  # sensible fallback

            # Compute optimum heights using your optimizer
            try:
                opt20 = float(self._yao.optimize_for_overpressure(yield_kt, 20))
            except Exception:
                opt20 = None
            try:
                opt5  = float(self._yao.optimize_for_overpressure(yield_kt, 5))
            except Exception:
                opt5 = None

            spec_len = max(major_len * 0.5, self.ring_thickness)*0.3333
            col20 = QColor("#d81b60")  # pink/red
            col5  = QColor("#1e88e5")  # blue

            # set a small gap beyond the tick end
            gap_px = max(4.0, 0.25 * self.ring_thickness)
            label_off = spec_len + gap_px + 15 # <-- label slightly beyond tick tip

            f = p.font()
            f.setBold(False)
            f.setPointSize(7)
            p.setFont(f)

            # 20 psi optimum (tick outward + label outside)
            if opt20 is not None and vmin <= opt20 <= vmax:
                self._draw_tick_at_value(p, c, r, opt20, vmin, vmax, spec_len, col20, outward=True)
                ang20 = self._value_to_angle_deg(opt20, vmin, vmax)

                try:
                    self._draw_text_on_ring(p, c, r, ang20, "20 psi", color=col20, offset_px=label_off)
                except TypeError:
                    # if your _draw_text_on_ring doesn't have offset_px yet, use your old call
                    self._draw_text_on_ring(p, c, r, ang20, "20 psi", color=col20)

            # 5 psi optimum (tick outward + label outside)
            if opt5 is not None and vmin <= opt5 <= vmax:
                self._draw_tick_at_value(p, c, r, opt5, vmin, vmax, spec_len, col5, outward=True)
                ang5 = self._value_to_angle_deg(opt5, vmin, vmax)
                try:
                    self._draw_text_on_ring(p, c, r, ang5, "5 psi", color=col5, offset_px=label_off)
                except TypeError:
                    self._draw_text_on_ring(p, c, r, ang5, "5 psi", color=col5)
        # ==============================================

        # Ring title and optional 0 marker (unchanged)
        self._draw_text_on_ring(p, c, r, self.start_angle_deg - 6, label)
        if include_zero:
            self._draw_tick_at_angle(p, c, r, self.start_angle_deg, self.ring_thickness, self.col_zero_label)
            self._draw_text_on_ring(p, c, r, self.start_angle_deg, '0', color=self.col_zero_label)

        p.restore()

    def _draw_tick_at_value(self, p, c, r, value, vmin, vmax, length, color, outward=False):
        ang = self._value_to_angle_deg(value, vmin, vmax)
        self._draw_tick_at_angle(p, c, r, ang, length, color, outward=outward)

    def _draw_tick_at_angle(self, p, c, r, ang_deg, length, color, outward=False):
        theta = math.radians(ang_deg)
        cs, sn = math.cos(theta), math.sin(theta)

        # outward=False (default): tick goes inward (r → r - length)
        # outward=True:            tick goes outward (r → r + length)
        r2 = (r - length) if not outward else (r + length)

        po = QPoint(int(c.x() + r  * cs), int(c.y() + r  * sn))  # on ring
        pi = QPoint(int(c.x() + r2 * cs), int(c.y() + r2 * sn))  # tick end

        p.setPen(QPen(color, max(2, int(self.ring_thickness * 0.2))))
        p.drawLine(po, pi)

    def _fmt_k(self, v: float) -> str:
        """1 -> '1', 10 -> '10', 100 -> '100', 1_000 -> '1k', 10_000 -> '10k', 100_000 -> '100k'."""
        v = float(v)
        if v >= 1000:
            return f"{int(v // 1000)}k"
        return str(int(v))        

    def _draw_text_on_ring(self, p, c, r, ang_deg, text, color=None, offset_px=None):
        if color is None:
            color = self.col_text
        if offset_px is None:
            offset_px = self.ring_thickness * 0.7  # tighter by default

        theta = math.radians(ang_deg)
        cs, sn = math.cos(theta), math.sin(theta)

        # Same convention as your ticks: y uses +sn
        x = float(c.x()) + (r + offset_px) * cs
        y = float(c.y()) + (r + offset_px) * sn

        f = p.font()
        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QPen(color, 1))

        fm = QFontMetricsF(p.font())
        w = fm.horizontalAdvance(text)
        h = fm.height()
        rect = QRectF(x - w / 2.0, y - h / 2.0, w, h)
        p.drawText(rect, Qt.AlignCenter, text)

    def _draw_indicator(self, p, c, r, value, vmin, vmax, color, zero_mode=False):
        ang = self._value_to_angle_deg(value, vmin, vmax)
        rad = math.radians(ang)
        s, cc = math.sin(rad), math.cos(rad)
        p_outer = QPoint(int(c.x() + r * cc), int(c.y() + r * s))
        p_inner = QPoint(int(c.x() + (r - self.indicator_length) * cc), int(c.y() + (r - self.indicator_length) * s))
        p.save()
        halo = QPen(QColor(255, 255, 255, 120), max(6, int(self.ring_thickness * 0.5)))
        halo.setCapStyle(Qt.RoundCap)
        p.setPen(halo)
        p.drawLine(p_inner, p_outer)
        stroke = QPen(color, max(3, int(self.ring_thickness * 0.35)))
        stroke.setCapStyle(Qt.RoundCap)
        p.setPen(stroke)
        p.drawLine(p_inner, p_outer)
        knob_r = max(8, int(self.ring_thickness * 0.45))
        p.setBrush(QBrush(QColor(255, 255, 255, 120)))
        p.setPen(QPen(QColor(0, 0, 0, 120), 1))
        p.drawEllipse(p_outer, knob_r, knob_r)
        p.setPen(QPen(color, 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(p_outer, knob_r, knob_r)
        base_w = max(8, int(self.ring_thickness * 0.6))
        head_h = max(6, int(self.ring_thickness * 0.5))
        bx = c.x() + (r - knob_r - 2) * cc
        by = c.y() + (r - knob_r - 2) * s
        ps, pc = math.sin(rad + math.pi / 2), math.cos(rad + math.pi / 2)
        p1 = QPoint(int(bx + (base_w / 2) * pc), int(by + (base_w / 2) * ps))
        p2 = QPoint(int(bx - (base_w / 2) * pc), int(by - (base_w / 2) * ps))
        p3 = QPoint(int(c.x() + (r + head_h) * cc), int(c.y() + (r + head_h) * s))
        path = QPainterPath()
        path.moveTo(p1)
        path.lineTo(p2)
        path.lineTo(p3)
        path.closeSubpath()
        p.setPen(QPen(QColor(255, 255, 255, 120), 1))
        p.setBrush(QBrush(color))
        p.drawPath(path)
        if zero_mode:
            p.setPen(QPen(self.col_zero_label, 1.5))
            p.setBrush(Qt.NoBrush)
            f = p.font()            
            f.setBold(False)
            f.setPointSize(8)
            p.setFont(f)            
            p.drawText(p_outer.x() + knob_r + 4, p_outer.y() - knob_r - 2, '0')
        p.restore()

    # ---- Math helpers ----
#    def _value_to_angle_deg(self, value, vmin, vmax):
#        value = self._clamp(value, vmin, vmax)
#        a = math.log10(value) - math.log10(vmin)
#        b = math.log10(vmax) - math.log10(vmin)
#        t = 0.0 if b == 0 else a / b
#        return self.start_angle_deg + t * self.sweep_deg
    
    def _value_to_angle_deg(self, v, vmin, vmax):
        # map log10(v) into [start_angle, start_angle + sweep]
        v = max(min(v, vmax), vmin)
        t = (math.log10(v) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))
        return self.start_angle_deg + t * self.sweep_deg    

    def _angle_to_value(self, ang_deg, vmin, vmax):
        t = (ang_deg - self.start_angle_deg) / self.sweep_deg
        t = self._clamp(t, 0.0, 1.0)
        logv = math.log10(vmin) + t * (math.log10(vmax) - math.log10(vmin))
        return 10 ** logv

    def _angle_from_point(self, pt, c):
        dx = pt.x() - c.x()
        dy = pt.y() - c.y()
        ang = math.degrees(math.atan2(dy, dx))
        start = self.start_angle_deg
        end = start + self.sweep_deg
        if ang < start:
            if abs((ang - start + 180) % 360 - 180) <= abs((ang - end + 180) % 360 - 180):
                ang = start
            else:
                ang = end
        elif ang > end:
            if abs((ang - start + 180) % 360 - 180) < abs((ang - end + 180) % 360 - 180):
                ang = start
            else:
                ang = end
        return ang

    # ---- Mouse ----
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        rect = self.rect()
        c = rect.center()
        size = min(rect.width(), rect.height())
        radius = int(size * self.outer_radius_ratio / 2) - self.pad
        ry = int(radius * self.yield_ring_ratio)
        rh = int(radius * self.height_ring_ratio)
        pos = e.pos()
        self._active_ring = 'yield' if abs(self._dist_to(c, pos) - ry) < abs(self._dist_to(c, pos) - rh) else 'height'
        self._update_value_from_pos(pos, c)

    def mouseMoveEvent(self, e):
        if self._active_ring is None:
            return
        self._update_value_from_pos(e.pos(), self.rect().center())

    def mouseReleaseEvent(self, e):
        self._active_ring = None

    def _update_value_from_pos(self, pos, c):
        ang = self._angle_from_point(pos, c)
        if self._active_ring == 'yield':
            val = int(round(self._angle_to_value(ang, self.yield_min, self.yield_max)))
            self.set_yield_value(val)
        elif self._active_ring == 'height':
            if abs(((ang - self.start_angle_deg + 180) % 360) - 180) < 2.0:
                self.set_height_value(0)
            else:
                val = int(round(self._angle_to_value(ang, self.height_min_positive, self.height_max)))
                self.set_height_value(val)

    def _dist_to(self, c, p):
        return ((p.x() - c.x()) ** 2 + (p.y() - c.y()) ** 2) ** 0.5
        
