"""
Digital Twin Agent for Coiltech Industrial Automation Platform.

Handles real-time sensor data interpretation, anomaly detection,
machine health monitoring, and predictive alerts for Coiltech
press feeding systems, decoilers, and CTL lines.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentContext

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a Coiltech machine sensor stream."""
    machine_code: str
    machine_name: str
    anomaly_type: str
    current_value: float
    threshold: float
    severity: str  # info, warning, critical, emergency
    timestamp: str
    recommendation: str


class DigitalTwinAgent(BaseAgent):
    """
    Digital twin agent for Coiltech press feeding systems.

    Monitors domain-specific sensor channels:
    - Servo load (%)
    - Feed length deviation (mm)
    - Line speed (m/min)
    - Coil remaining diameter (mm)
    - Bearing temperature (C)
    - Vibration (mm/s)
    - Air pressure (bar)
    - Vacuum status
    - Cycle count
    - Lubrication status
    - Encoder drift (mm)
    - Power consumption (kW)
    """

    def __init__(self, **kwargs):
        from config.settings import get_settings
        settings = get_settings()

        super().__init__(
            name="DigitalTwinAgent",
            department="digital_twin",
            description=(
                "Monitors real-time sensor data from Coiltech machines in the field. "
                "Detects anomalies for servo load, feed deviation, vibration, bearing "
                "temperature, air pressure, and other domain-specific parameters."
            ),
            **kwargs,
        )

        self.thresholds = {
            "servo_load_pct": settings.SENSOR_SERVO_LOAD_THRESHOLD,        # 85%
            "feed_deviation_mm": settings.SENSOR_FEED_DEVIATION_THRESHOLD,  # 0.15 mm
            "bearing_temperature_c": settings.SENSOR_BEARING_TEMP_THRESHOLD,  # 75 C
            "vibration_mm_s": settings.SENSOR_VIBRATION_THRESHOLD,          # 7.5 mm/s
            "air_pressure_min": settings.SENSOR_AIR_PRESSURE_MIN,           # 4.5 bar
            "air_pressure_max": settings.SENSOR_AIR_PRESSURE_MAX,           # 7.0 bar
        }

        self._queries = {
            "latest_readings": """
                SELECT ss.*, ib.code AS machine_code, ib.machine_name,
                       c.name AS customer_name
                FROM sensor_streams ss
                JOIN installed_base ib ON ss.machine_id = ib.id
                JOIN customers c ON ib.customer_id = c.id
                WHERE ss.id IN (
                    SELECT MAX(id) FROM sensor_streams GROUP BY machine_id
                )
                ORDER BY ib.code
            """,
            "active_alerts": "SELECT * FROM v_active_alerts",
            "recent_alerts": """
                SELECT ma.alert_type, ma.severity, ma.value, ma.threshold,
                       ma.message, ma.timestamp, ma.acknowledged,
                       ib.code AS machine_code, ib.machine_name,
                       c.name AS customer_name
                FROM machine_alerts ma
                JOIN installed_base ib ON ma.machine_id = ib.id
                JOIN customers c ON ib.customer_id = c.id
                WHERE ma.timestamp >= DATETIME('now', '-48 hours')
                ORDER BY ma.timestamp DESC LIMIT 30
            """,
            "machine_health": """
                SELECT ib.code, ib.machine_name, ib.status,
                       c.name AS customer_name,
                       COUNT(CASE WHEN ma.resolved = 0 THEN 1 END) AS open_alerts,
                       COUNT(CASE WHEN ma.severity = 'critical' AND ma.resolved = 0 THEN 1 END) AS critical_alerts,
                       MAX(ss.servo_load_pct) AS max_servo_load,
                       AVG(ss.bearing_temperature_c) AS avg_bearing_temp,
                       MAX(ss.vibration_mm_s) AS max_vibration
                FROM installed_base ib
                JOIN customers c ON ib.customer_id = c.id
                LEFT JOIN machine_alerts ma ON ib.id = ma.machine_id
                LEFT JOIN sensor_streams ss ON ib.id = ss.machine_id
                    AND ss.timestamp >= DATETIME('now', '-24 hours')
                WHERE ib.status = 'operational'
                GROUP BY ib.id
                ORDER BY open_alerts DESC, critical_alerts DESC
            """,
            "alert_trends": """
                SELECT DATE(timestamp) AS date, alert_type, severity,
                       COUNT(*) AS count
                FROM machine_alerts
                WHERE timestamp >= DATE('now', '-7 days')
                GROUP BY DATE(timestamp), alert_type, severity
                ORDER BY date DESC
            """,
            "line_runs_recent": """
                SELECT lr.run_date, lr.shift, lr.total_strokes, lr.target_strokes,
                       lr.good_parts, lr.reject_parts, lr.downtime_minutes,
                       lr.avg_speed_spm, lr.coils_processed,
                       ib.code AS machine_code, ib.machine_name
                FROM line_runs lr
                JOIN installed_base ib ON lr.machine_id = ib.id
                ORDER BY lr.run_date DESC, lr.shift LIMIT 20
            """,
        }

    async def handle(self, query: str, context: Optional[AgentContext] = None) -> AgentResponse:
        logger.info("DigitalTwinAgent handling query", query=query[:60])

        # Fetch all sensor-related data
        latest = await self.query_database(self._queries["latest_readings"])
        active_alerts = await self.query_database(self._queries["active_alerts"])
        recent_alerts = await self.query_database(self._queries["recent_alerts"])
        health = await self.query_database(self._queries["machine_health"])
        trends = await self.query_database(self._queries["alert_trends"])
        line_runs = await self.query_database(self._queries["line_runs_recent"])
        focused_latest = self._focus_records_by_query(
            query,
            latest,
            ["machine_code", "machine_name", "customer_name"],
            max_items=8,
        )
        focused_alerts = self._focus_records_by_query(
            query,
            active_alerts + recent_alerts,
            ["machine_code", "machine_name", "customer_name", "alert_type", "severity", "message"],
            max_items=10,
        )
        focused_health = self._focus_records_by_query(
            query,
            health,
            ["code", "machine_name", "customer_name", "status"],
            max_items=8,
        )

        # Run anomaly detection on latest readings
        anomalies = self._detect_anomalies(latest) if latest else []

        # Generate system state narrative
        state_narrative = self._generate_state_narrative(latest, anomalies, health)

        rag_context = ""
        sources = []
        if context and context.user_metadata:
            rag_context = context.user_metadata.get("rag_context", "")
            sources = context.user_metadata.get("rag_sources", [])

        sql_result = None
        if self.should_attempt_nl_to_sql(query):
            try:
                sql_result = await self.nl_to_sql(query)
            except Exception:
                pass

        data_context = f"""
{state_narrative}

FOCUSED SENSOR READINGS FOR THIS QUERY:
{self._format_table(focused_latest)}

FOCUSED ALERTS FOR THIS QUERY:
{self._format_table(focused_alerts)}

FOCUSED MACHINE HEALTH FOR THIS QUERY:
{self._format_table(focused_health)}

ACTIVE ALERTS (unresolved):
{self._format_table(active_alerts[:15])}

RECENT ALERTS (48h):
{self._format_table(recent_alerts[:15])}

MACHINE HEALTH OVERVIEW:
{self._format_table(health[:15])}

ALERT TRENDS (7 days):
{self._format_table(trends[:15])}

RECENT LINE RUNS:
{self._format_table(line_runs[:10])}
"""
        if sql_result and sql_result.get("success"):
            data_context += f"\nSPECIFIC QUERY RESULTS:\nSQL: {sql_result.get('generated_sql')}\n{self._format_table(sql_result.get('results', [])[:15])}\n"

        if rag_context:
            data_context += f"\nSERVICE DOCUMENTATION:\n{rag_context}\n"

        prompt = self.build_prompt(query=query, context=context, additional_context=data_context)

        system_prompt = self._system_prompt + """
When interpreting digital twin / sensor data for Coiltech machines:
1. Prioritize safety-critical anomalies (servo overload, high vibration, air pressure loss)
2. Explain sensor readings in the context of press feeding operations
3. Feed deviation impacts part quality — correlate with reject counts if available
4. Servo load trending upward may indicate mechanical wear, material change, or alignment issue
5. Bearing temperature rise signals lubrication or alignment problems
6. Low air pressure affects gripper/transfer/pilot release reliability
7. Provide actionable recommendations: immediate, short-term, and long-term
8. Reference specific machine codes (IB-XXXX) and customer names
9. For line run data, calculate OEE when possible (availability x performance x quality)
10. Suggest creating a service ticket for persistent or critical anomalies
11. If focused alerts/readings are present, answer from them first
12. Do not invent real-time values not present in the sensor/alert data
13. Respond in the same language as the user"""

        response = await self.call_llm(prompt, system_prompt)

        # Determine overall state
        state = "normal"
        if any(a.severity == "emergency" for a in anomalies):
            state = "emergency"
        elif any(a.severity == "critical" for a in anomalies):
            state = "critical"
        elif any(a.severity == "warning" for a in anomalies):
            state = "warning"

        confidence = 0.9 if latest else 0.5

        return self.format_response(
            content=response,
            data={
                "state": state,
                "anomaly_count": len(anomalies),
                "anomalies": [
                    {
                        "machine_code": a.machine_code,
                        "type": a.anomaly_type,
                        "value": a.current_value,
                        "threshold": a.threshold,
                        "severity": a.severity,
                        "recommendation": a.recommendation,
                    }
                    for a in anomalies
                ],
                "active_alert_count": len(active_alerts),
                "focused_latest": focused_latest,
                "focused_alerts": focused_alerts,
                "focused_health": focused_health,
                "sql_results": sql_result.get("results") if sql_result else None,
            },
            sources=sources,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def _detect_anomalies(self, readings: List[Dict]) -> List[AnomalyResult]:
        anomalies = []
        for r in readings:
            machine_code = r.get("machine_code", "unknown")
            machine_name = r.get("machine_name", "unknown")
            ts = str(r.get("timestamp", ""))

            # Servo overload
            servo = r.get("servo_load_pct")
            if servo is not None and servo > self.thresholds["servo_load_pct"]:
                sev = self._severity(servo, self.thresholds["servo_load_pct"], 100)
                anomalies.append(AnomalyResult(
                    machine_code=machine_code, machine_name=machine_name,
                    anomaly_type="servo_overload", current_value=servo,
                    threshold=self.thresholds["servo_load_pct"], severity=sev,
                    timestamp=ts,
                    recommendation=self._recommend("servo_overload", sev),
                ))

            # Feed deviation
            actual = r.get("feed_length_actual_mm")
            target = r.get("feed_length_target_mm")
            if actual is not None and target is not None and target > 0:
                deviation = abs(actual - target)
                if deviation > self.thresholds["feed_deviation_mm"]:
                    sev = self._severity(deviation, self.thresholds["feed_deviation_mm"], 0.5)
                    anomalies.append(AnomalyResult(
                        machine_code=machine_code, machine_name=machine_name,
                        anomaly_type="feed_deviation", current_value=deviation,
                        threshold=self.thresholds["feed_deviation_mm"], severity=sev,
                        timestamp=ts,
                        recommendation=self._recommend("feed_deviation", sev),
                    ))

            # Bearing temperature
            bt = r.get("bearing_temperature_c")
            if bt is not None and bt > self.thresholds["bearing_temperature_c"]:
                sev = self._severity(bt, self.thresholds["bearing_temperature_c"], 95)
                anomalies.append(AnomalyResult(
                    machine_code=machine_code, machine_name=machine_name,
                    anomaly_type="high_bearing_temp", current_value=bt,
                    threshold=self.thresholds["bearing_temperature_c"], severity=sev,
                    timestamp=ts,
                    recommendation=self._recommend("bearing_temp", sev),
                ))

            # Vibration
            vib = r.get("vibration_mm_s")
            if vib is not None and vib > self.thresholds["vibration_mm_s"]:
                sev = self._severity(vib, self.thresholds["vibration_mm_s"], 15)
                anomalies.append(AnomalyResult(
                    machine_code=machine_code, machine_name=machine_name,
                    anomaly_type="high_vibration", current_value=vib,
                    threshold=self.thresholds["vibration_mm_s"], severity=sev,
                    timestamp=ts,
                    recommendation=self._recommend("vibration", sev),
                ))

            # Air pressure (too low or too high)
            ap = r.get("air_pressure_bar")
            if ap is not None:
                if ap < self.thresholds["air_pressure_min"]:
                    deficit = self.thresholds["air_pressure_min"] - ap
                    sev = "critical" if deficit > 1.5 else ("warning" if deficit > 0.5 else "info")
                    anomalies.append(AnomalyResult(
                        machine_code=machine_code, machine_name=machine_name,
                        anomaly_type="low_air_pressure", current_value=ap,
                        threshold=self.thresholds["air_pressure_min"], severity=sev,
                        timestamp=ts,
                        recommendation=self._recommend("low_air_pressure", sev),
                    ))
                elif ap > self.thresholds["air_pressure_max"]:
                    sev = "warning"
                    anomalies.append(AnomalyResult(
                        machine_code=machine_code, machine_name=machine_name,
                        anomaly_type="high_air_pressure", current_value=ap,
                        threshold=self.thresholds["air_pressure_max"], severity=sev,
                        timestamp=ts,
                        recommendation=self._recommend("high_air_pressure", sev),
                    ))

            # Lubrication overdue
            lub = r.get("lubrication_ok")
            if lub is not None and lub == 0:
                anomalies.append(AnomalyResult(
                    machine_code=machine_code, machine_name=machine_name,
                    anomaly_type="lubrication_overdue", current_value=0,
                    threshold=1, severity="warning", timestamp=ts,
                    recommendation="Schedule lubrication service. Check grease levels and auto-lube unit operation.",
                ))

            # Encoder drift
            drift = r.get("encoder_drift_mm")
            if drift is not None and abs(drift) > 0.1:
                sev = "critical" if abs(drift) > 0.3 else "warning"
                anomalies.append(AnomalyResult(
                    machine_code=machine_code, machine_name=machine_name,
                    anomaly_type="encoder_drift", current_value=abs(drift),
                    threshold=0.1, severity=sev, timestamp=ts,
                    recommendation=self._recommend("encoder_drift", sev),
                ))

        return anomalies

    def _severity(self, value: float, threshold: float, critical_value: float) -> str:
        excess_pct = ((value - threshold) / threshold) * 100 if threshold else 0
        if value >= critical_value or excess_pct > 30:
            return "critical"
        elif excess_pct > 15:
            return "warning"
        return "info"

    def _recommend(self, anomaly_type: str, severity: str) -> str:
        recs = {
            "servo_overload": {
                "critical": "IMMEDIATE: Reduce press speed or stop line. Check material thickness, roller alignment, and servo drive parameters.",
                "warning": "Monitor servo current trend. Verify material is within specification. Schedule drive inspection.",
                "info": "Note for next PM. Check belt tension and roller condition.",
            },
            "feed_deviation": {
                "critical": "STOP LINE: Feed accuracy compromised. Check encoder, roller pressure, pilot release timing, and material slippage.",
                "warning": "Check roller condition and pressure settings. Verify encoder calibration. Monitor part quality.",
                "info": "Minor deviation — monitor trend and check at next setup change.",
            },
            "bearing_temp": {
                "critical": "IMMEDIATE: Stop machine, check bearing condition and lubrication. Risk of seizure.",
                "warning": "Schedule bearing inspection. Verify grease type and quantity. Check alignment.",
                "info": "Temperature elevated — ensure adequate cooling and lubrication.",
            },
            "vibration": {
                "critical": "IMMEDIATE: Stop machine. Inspect bearings, shaft alignment, coupling, and foundation bolts.",
                "warning": "Schedule vibration analysis. Check bearing condition and balancing.",
                "info": "Slight increase in vibration — monitor trend.",
            },
            "low_air_pressure": {
                "critical": "STOP LINE: Insufficient air pressure for reliable gripper/transfer operation. Check compressor and lines for leaks.",
                "warning": "Air pressure dropping — check filters, regulator, and drain traps. Verify compressor output.",
                "info": "Air pressure slightly low — monitor and check FRL unit.",
            },
            "high_air_pressure": {
                "warning": "Reduce pressure setting. Check regulator. Risk of seal/cylinder damage.",
                "info": "Slightly above spec — adjust regulator.",
            },
            "encoder_drift": {
                "critical": "STOP: Encoder drift exceeds tolerance. Recalibrate or replace encoder. Check coupling and shaft.",
                "warning": "Encoder drifting — schedule calibration check. Verify encoder coupling is tight.",
                "info": "Minor drift — note for next service.",
            },
        }
        return recs.get(anomaly_type, {}).get(severity, "Monitor and assess.")

    # ------------------------------------------------------------------
    # State narrative
    # ------------------------------------------------------------------
    def _generate_state_narrative(
        self,
        readings: List[Dict],
        anomalies: List[AnomalyResult],
        health: List[Dict],
    ) -> str:
        lines = ["COILTECH DIGITAL TWIN — SYSTEM STATE"]
        lines.append("=" * 50)
        lines.append(f"Report Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        crit = sum(1 for a in anomalies if a.severity == "critical")
        warn = sum(1 for a in anomalies if a.severity == "warning")

        if crit:
            lines.append(f"\nSTATUS: CRITICAL ({crit} critical anomalies)")
        elif warn:
            lines.append(f"\nSTATUS: WARNING ({warn} warnings)")
        elif anomalies:
            lines.append(f"\nSTATUS: INFO ({len(anomalies)} items to note)")
        else:
            lines.append("\nSTATUS: NORMAL — All parameters within limits")

        if health:
            lines.append("\nMACHINE HEALTH:")
            for h in health[:10]:
                oa = h.get("open_alerts", 0)
                tag = "ALERT" if oa else "OK"
                lines.append(
                    f"  [{tag}] {h.get('code','?')} {h.get('machine_name','?')}: "
                    f"{h.get('customer_name','?')} — open alerts: {oa}"
                )

        if anomalies:
            lines.append("\nDETECTED ANOMALIES:")
            for a in anomalies[:10]:
                lines.append(
                    f"  [{a.severity.upper()}] {a.machine_code}: {a.anomaly_type} "
                    f"({a.current_value:.2f} vs threshold {a.threshold})"
                )

        if readings:
            lines.append("\nLATEST SENSOR READINGS:")
            for r in readings[:8]:
                parts = [f"{r.get('machine_code','?')} ({r.get('machine_name','?')})"]
                if r.get("servo_load_pct") is not None:
                    parts.append(f"Servo={r['servo_load_pct']:.1f}%")
                if r.get("bearing_temperature_c") is not None:
                    parts.append(f"BearTemp={r['bearing_temperature_c']:.1f}C")
                if r.get("vibration_mm_s") is not None:
                    parts.append(f"Vib={r['vibration_mm_s']:.2f}mm/s")
                if r.get("air_pressure_bar") is not None:
                    parts.append(f"Air={r['air_pressure_bar']:.1f}bar")
                if r.get("line_speed_mpm") is not None:
                    parts.append(f"Speed={r['line_speed_mpm']:.1f}m/min")
                lines.append("  " + " | ".join(parts))

        lines.append("\nTHRESHOLD SETTINGS:")
        lines.append(f"  Servo Load: {self.thresholds['servo_load_pct']}%")
        lines.append(f"  Feed Deviation: {self.thresholds['feed_deviation_mm']} mm")
        lines.append(f"  Bearing Temp: {self.thresholds['bearing_temperature_c']} C")
        lines.append(f"  Vibration: {self.thresholds['vibration_mm_s']} mm/s")
        lines.append(f"  Air Pressure: {self.thresholds['air_pressure_min']}-{self.thresholds['air_pressure_max']} bar")

        return "\n".join(lines)

    def _format_table(self, data: List[Dict]) -> str:
        if not data:
            return "No data available"
        headers = list(data[0].keys())
        rows = [" | ".join(str(item.get(h, ""))[:25] for h in headers) for item in data]
        header_row = " | ".join(h[:25] for h in headers)
        return f"{header_row}\n{'-' * len(header_row)}\n" + "\n".join(rows)
