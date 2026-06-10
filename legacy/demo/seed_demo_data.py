"""
Seed realistic demo data for the Coiltech Industrial Automation Platform.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import SQLiteManager
from rag.rag_pipeline import RAGPipeline


DEMO_DOCUMENTS = [
    {
        "file_name": "compact_drive_selection_guide.txt",
        "collection": "products_compact_drives",
        "title": "Compact Drive Selection Guide",
        "doc_type": "datasheet",
        "content": """
Compact Drive Selection Guide

CD-300 is intended for compact feed lines up to 300 mm strip width, 3.0 mm mild steel,
and 35 m/min line speed. Typical applications include appliance panels and medium-size stampings.

CD-600 is recommended when the customer needs higher coil weight, wider strip, and servo feed
accuracy under fast die change conditions. Standard scope includes hydraulic decoiler interface,
loop control, pilot release, and feed roll wear monitoring.
""",
    },
    {
        "file_name": "servo_drive_alarm_matrix.txt",
        "collection": "service_maintenance",
        "title": "Servo Drive Alarm Matrix",
        "doc_type": "service_bulletin",
        "content": """
Servo Drive Alarm Matrix

Alarm E-101: Servo overload.
Primary checks:
1. Strip thickness matches job recipe.
2. Feed roll pressure is not excessive.
3. Encoder coupling is tight.
4. Straightener entry angle is correct.

Alarm E-214: Encoder drift.
Primary checks:
1. Check backlash in coupling.
2. Verify reference mark at home position.
3. Recalibrate feed length after coil change.
""",
    },
    {
        "file_name": "ctl_commissioning_checklist.txt",
        "collection": "installation_commissioning",
        "title": "CTL Line Commissioning Checklist",
        "doc_type": "commissioning_checklist",
        "content": """
Cut-to-Length Line Commissioning Checklist

Mechanical:
- Foundation bolts torqued and marked
- Straightener alignment verified
- Shear blade gap adjusted to material thickness

Electrical and controls:
- Emergency stop chain tested
- Line encoder scaling verified
- Recipe transfer from HMI to PLC confirmed

Process validation:
- Feed length deviation below 0.15 mm
- Line speed ramp stable to rated speed
- Stacker discharge synchronized with shear cycle
""",
    },
    {
        "file_name": "spare_parts_critical_stock.txt",
        "collection": "spare_parts",
        "title": "Critical Spare Parts Stocking Guide",
        "doc_type": "manual",
        "content": """
Critical Spare Parts Stocking Guide

Recommended customer stock for high-uptime lines:
- Servo motor SM-400: 1 piece for each two active compact drives
- Encoder kit ENC-CTL-01: 2 pieces per CTL line
- Bearing set BRG-6208-2RS: 4 pieces per straightener section
- Vacuum valve kit VAC-24V-06: 2 pieces per transfer robot cell
""",
    },
    {
        "file_name": "quotation_terms_2026.txt",
        "collection": "sales_and_quotations",
        "title": "Commercial Terms 2026",
        "doc_type": "manual",
        "content": """
Commercial Terms 2026

Standard quotation validity: 30 days
Default incoterm: EXW Bursa
Standard payment terms for strategic customers:
- 40 percent advance
- 50 percent before shipment
- 10 percent after SAT sign-off
""",
    },
    {
        "file_name": "ce_compliance_press_lines.txt",
        "collection": "quality_documents",
        "title": "CE Compliance for Press Feeding Lines",
        "doc_type": "certificate",
        "content": """
CE Compliance for Press Feeding and Transfer Lines

Applicable references:
- Machinery Directive 2006/42/EC
- EN ISO 12100 risk assessment
- EN 60204-1 electrical safety
- EN ISO 13849-1 safety-related control systems
""",
    },
    {
        "file_name": "special_project_destacker_layout.txt",
        "collection": "special_projects",
        "title": "Destacker and Blank Line Integration Concept",
        "doc_type": "drawing",
        "content": """
Destacker and Blank Line Integration Concept

Project concept:
- Destacker loads blanks into transfer conveyor
- Conveyor synchronizes with press transfer robot cell
- Blank counter and double-sheet sensor report to line PLC
- Optional reject gate diverts oil-contaminated blanks
""",
    },
]


def reset_database(db: SQLiteManager) -> None:
    conn = db.connect()
    conn.executescript(
        """
        PRAGMA foreign_keys = OFF;
        DELETE FROM uploaded_documents;
        DELETE FROM technical_documents;
        DELETE FROM quality_events;
        DELETE FROM line_runs;
        DELETE FROM machine_alerts;
        DELETE FROM sensor_streams;
        DELETE FROM spare_part_stock;
        DELETE FROM spare_parts;
        DELETE FROM maintenance_logs;
        DELETE FROM maintenance_plans;
        DELETE FROM service_tickets;
        DELETE FROM commissioning_reports;
        DELETE FROM installed_base;
        DELETE FROM bom_components;
        DELETE FROM line_configurations;
        DELETE FROM project_machines;
        DELETE FROM project_milestones;
        DELETE FROM projects;
        DELETE FROM quotation_items;
        DELETE FROM quotations;
        DELETE FROM project_opportunities;
        DELETE FROM machine_capabilities;
        DELETE FROM product_models;
        DELETE FROM product_families;
        DELETE FROM contacts;
        DELETE FROM customers;
        DELETE FROM sqlite_sequence;
        PRAGMA foreign_keys = ON;
        """
    )
    conn.commit()


def seed_relational_data(db: SQLiteManager) -> None:
    conn = db.connect()
    cursor = conn.cursor()

    customers = [
        ("CUST-001", "Ford Otosan", "TR", "Kocaeli", "automotive", "strategic", "https://www.fordotosan.com.tr", "High-volume stamping customer"),
        ("CUST-002", "Tofas", "TR", "Bursa", "automotive", "strategic", "https://www.tofas.com.tr", "Press shop modernization program"),
        ("CUST-003", "Arcelik", "TR", "Eskisehir", "appliance", "key", "https://www.arcelik.com", "Appliance panel feed lines"),
        ("CUST-004", "Vestel", "TR", "Manisa", "appliance", "key", "https://www.vestel.com.tr", "CTL and press feed opportunities"),
        ("CUST-005", "Beko Romania", "RO", "Gaesti", "appliance", "standard", "https://www.beko.com", "Installed base of compact drives"),
        ("CUST-006", "Durmazlar", "TR", "Bursa", "general_stamping", "prospect", "https://www.durmazlar.com.tr", "Prospect for custom blank handling"),
    ]
    cursor.executemany(
        """
        INSERT INTO customers (code, name, country, city, industry, tier, website, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        customers,
    )

    contacts = [
        (1, "Mert Yilmaz", "Press Shop Manager", "mert.yilmaz@fordotosan.com", "+90-262-000-1001", "engineering", 1),
        (2, "Selin Acar", "Maintenance Chief", "selin.acar@tofas.com.tr", "+90-224-000-2002", "maintenance", 1),
        (3, "Onur Demir", "Purchasing Specialist", "onur.demir@arcelik.com", "+90-222-000-3003", "purchasing", 1),
        (4, "Emre Kaya", "Project Engineer", "emre.kaya@vestel.com.tr", "+90-236-000-4004", "engineering", 1),
        (5, "Ana Popescu", "Plant Manager", "ana.popescu@beko.com", "+40-245-000-5005", "management", 1),
        (6, "Kerem Polat", "Production Director", "kerem.polat@durmazlar.com.tr", "+90-224-000-6006", "management", 1),
    ]
    cursor.executemany(
        """
        INSERT INTO contacts (customer_id, name, title, email, phone, department, is_primary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        contacts,
    )

    product_families = [
        ("PF-PFS", "Press Feeding Systems", "press_feeding", "Complete press feeding lines with decoiler, straightener and feeder."),
        ("PF-CD", "Compact Drives", "press_feeding", "Compact servo feed lines for limited footprint applications."),
        ("PF-LD", "Loop Type Drives", "press_feeding", "Loop type feed systems for higher press speeds."),
        ("PF-BL", "Blank Lines", "blank_line", "Blanking and blank transfer systems."),
        ("PF-DS", "Destacker Systems", "transfer", "Destackers for blank separation and feeding."),
        ("PF-PTS", "Press Transfer Systems", "transfer", "Mechanical and servo press transfer systems."),
        ("PF-PTR", "Press Transfer Robots", "transfer", "Robot-based press transfer solutions."),
        ("PF-SD", "Servo Drives", "servo", "Servo feed and motion control families."),
        ("PF-TGSD", "Triggered / Gearbox Servo Drives", "servo", "Triggered and gearbox servo feed solutions."),
        ("PF-STSD", "Straightener Servo Drives", "servo", "Servo straightener feeds."),
        ("PF-ZSSD", "Zigzag Servo Drives", "servo", "Zigzag servo feeding for blank saving."),
        ("PF-DEC", "Decoilers / Uncoilers", "decoiler", "Decoiler and uncoiler families."),
        ("PF-MDEC", "Mechanical Decoilers", "decoiler", "Mechanical decoilers."),
        ("PF-HDEC", "Hydraulic Decoilers", "decoiler", "Hydraulic expansion decoilers."),
        ("PF-DMDEC", "Double Mandrel Decoilers", "decoiler", "Double mandrel decoilers for high uptime."),
        ("PF-PDEC", "Pallet Type Decoilers", "decoiler", "Pallet type decoilers for flexible coil change."),
        ("PF-CTL", "Cut-to-Length Lines", "ctl_line", "Cut-to-length line families."),
        ("PF-SCTL", "Standard Cut-to-Length Lines", "ctl_line", "Standard CTL lines."),
        ("PF-HPCTL", "High Performance Cut-to-Length Lines", "ctl_line", "High-speed CTL lines."),
        ("PF-SP", "Custom / Special Projects", "special", "Customer-specific integrated lines."),
    ]
    cursor.executemany(
        "INSERT INTO product_families (code, name, category, description) VALUES (?, ?, ?, ?)",
        product_families,
    )

    cursor.execute("SELECT id, code FROM product_families")
    family_ids = {code: family_id for family_id, code in cursor.fetchall()}

    product_models = [
        (family_ids["PF-CD"], "CD-300", "Compact Drive 300", 300, 3.0, 350, 35, 5000, 1500, 250, 7.5, 2200, "Compact feed line for appliance and light automotive parts."),
        (family_ids["PF-CD"], "CD-600", "Compact Drive 600", 600, 4.5, 500, 45, 8000, 1800, 400, 11.0, 3400, "Compact drive with larger coil and strip capacity."),
        (family_ids["PF-LD"], "LD-800", "Loop Drive 800", 800, 6.0, 600, 60, 12000, 2000, 630, 18.5, 5100, "Loop type feed line for medium and large press shops."),
        (family_ids["PF-ZSSD"], "ZZ-1300", "Zigzag Servo 1300", 1300, 2.5, 750, 40, 10000, 1800, 350, 15.0, 4800, "Zigzag servo feed for blank saving applications."),
        (family_ids["PF-MDEC"], "MD-5T", "Mechanical Decoiler 5T", 650, 4.0, None, 30, 5000, 1600, None, 3.0, 1900, "Mechanical decoiler for compact lines."),
        (family_ids["PF-HDEC"], "HD-10T", "Hydraulic Decoiler 10T", 1000, 6.0, None, 40, 10000, 2000, None, 5.5, 3600, "Hydraulic decoiler with powered mandrel expansion."),
        (family_ids["PF-DMDEC"], "DMD-20T", "Double Mandrel Decoiler 20T", 1600, 8.0, None, 55, 20000, 2200, None, 11.0, 6900, "Double mandrel decoiler for high uptime lines."),
        (family_ids["PF-PTR"], "PTR-6A", "Press Transfer Robot 6 Axis", None, None, None, 18, None, None, None, 12.0, 1600, "Six-axis robot for press transfer and destacker interface."),
        (family_ids["PF-SCTL"], "CTL-1250", "Standard CTL 1250", 1250, 4.0, 3000, 35, 12000, 2000, None, 30.0, 12000, "Standard cut-to-length line for appliance and service center use."),
        (family_ids["PF-HPCTL"], "CTL-1600HP", "High Performance CTL 1600", 1600, 6.0, 4000, 80, 20000, 2200, None, 55.0, 19000, "High-speed CTL line with dynamic shear."),
        (family_ids["PF-DS"], "DSK-800", "Destacker 800", 800, None, None, 20, None, None, None, 4.0, 2600, "Destacker for pre-cut blanks."),
        (family_ids["PF-TGSD"], "TG-500", "Triggered Gearbox Servo 500", 500, 5.0, 450, 50, 6000, 1700, None, 9.0, 2800, "Triggered servo feed for synchronized press lines."),
    ]
    cursor.executemany(
        """
        INSERT INTO product_models (
            family_id, model_code, name, max_strip_width_mm, max_strip_thickness_mm,
            max_feed_length_mm, max_speed_mpm, max_coil_weight_kg, max_coil_od_mm,
            press_force_ton, servo_power_kw, weight_kg, description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        product_models,
    )

    cursor.execute("SELECT id, model_code FROM product_models")
    model_ids = {model_code: model_id for model_id, model_code in cursor.fetchall()}

    capabilities = [
        (model_ids["CD-300"], "pilot_release", "yes", None, "Standard pneumatic pilot release"),
        (model_ids["CD-300"], "loop_control", "yes", None, "Integrated loop control"),
        (model_ids["CD-600"], "recipe_management", "yes", None, "Stores up to 100 recipes"),
        (model_ids["LD-800"], "high_speed_mode", "60", "mpm", "Rated with thin gauge steel"),
        (model_ids["ZZ-1300"], "zigzag_mode", "yes", None, "Blank saving mode"),
        (model_ids["HD-10T"], "mandrel_expansion", "hydraulic", None, "Hydraulic expansion"),
        (model_ids["DMD-20T"], "coil_change", "flying", None, "Automatic coil preparation"),
        (model_ids["PTR-6A"], "double_sheet_detection", "yes", None, "Blank pick validation"),
        (model_ids["CTL-1600HP"], "dynamic_shear", "yes", None, "Flying shear configuration"),
        (model_ids["TG-500"], "press_sync", "cam_signal", None, "Synchronized triggered feed"),
    ]
    cursor.executemany(
        "INSERT INTO machine_capabilities (model_id, capability_name, capability_value, unit, notes) VALUES (?, ?, ?, ?, ?)",
        capabilities,
    )

    opportunities = [
        ("OPP-2026-001", 1, "Ford Otosan High-Speed Blank Line", "Blank line and transfer interface for body panel production.", "proposal", 1450000, "EUR", 65, "2026-06-15", "Serkan Erdem", "referral", "Includes destacker and robot transfer options."),
        ("OPP-2026-002", 3, "Arcelik Compact Feed Line", "Compact drive line for white goods side panel production.", "negotiation", 420000, "EUR", 80, "2026-05-20", "Aylin Kose", "repeat_business", "Customer requests fast delivery."),
        ("OPP-2026-003", 4, "Vestel CTL Upgrade", "Replace old CTL line with higher performance model.", "qualification", 980000, "EUR", 50, "2026-07-10", "Bora Cetin", "website", "Upgrade with dynamic shear and stacker."),
        ("OPP-2026-004", 6, "Durmazlar Destacker Integration", "Special project to integrate destacker with existing press line.", "lead", 310000, "EUR", 30, "2026-08-01", "Bora Cetin", "exhibition", "Custom project layout needed."),
    ]
    cursor.executemany(
        """
        INSERT INTO project_opportunities (
            code, customer_id, title, description, stage, estimated_value, currency,
            probability_pct, expected_close_date, owner, source, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        opportunities,
    )

    quotations = [
        ("QUO-2026-001", 2, 3, 2, "under_review", 438000, "EUR", 30, 18, "40/50/10", "EXW", "Compact line with optional remote support.", "Aylin Kose", "2026-04-05"),
        ("QUO-2026-002", 1, 1, 1, "sent", 1495000, "EUR", 30, 32, "30/60/10", "EXW", "Includes destacker and transfer robot option.", "Serkan Erdem", "2026-04-11"),
        ("QUO-2025-014", 1, 2, 1, "accepted", 865000, "EUR", 30, 24, "40/50/10", "EXW", "Press transfer upgrade package.", "Serkan Erdem", "2025-09-18"),
    ]
    cursor.executemany(
        """
        INSERT INTO quotations (
            code, opportunity_id, customer_id, revision, status, total_amount, currency,
            validity_days, delivery_weeks, payment_terms, incoterm, notes, prepared_by, issued_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        quotations,
    )

    cursor.execute("SELECT id, code FROM quotations")
    quotation_ids = {code: quotation_id for quotation_id, code in cursor.fetchall()}

    quotation_items = [
        (quotation_ids["QUO-2026-001"], 10, model_ids["CD-300"], "Compact Drive 300 feed line", 1, 215000, 0, 215000, "Main line"),
        (quotation_ids["QUO-2026-001"], 20, model_ids["MD-5T"], "Mechanical Decoiler 5T", 1, 46000, 0, 46000, "Entry decoiler"),
        (quotation_ids["QUO-2026-001"], 30, model_ids["TG-500"], "Triggered Gearbox Servo 500", 1, 54000, 0, 54000, "Press synchronization"),
        (quotation_ids["QUO-2026-002"], 10, model_ids["DSK-800"], "Destacker 800", 1, 125000, 0, 125000, "Blank destacking"),
        (quotation_ids["QUO-2026-002"], 20, model_ids["PTR-6A"], "Press Transfer Robot 6 Axis", 2, 185000, 3, 358900, "Robot transfer cell"),
        (quotation_ids["QUO-2026-002"], 30, model_ids["LD-800"], "Loop Drive 800", 1, 285000, 0, 285000, "Feed line"),
        (quotation_ids["QUO-2025-014"], 10, model_ids["LD-800"], "Loop Drive 800", 1, 285000, 0, 285000, "Main feed system"),
        (quotation_ids["QUO-2025-014"], 20, model_ids["PTR-6A"], "Press Transfer Robot 6 Axis", 1, 185000, 0, 185000, "Transfer robot"),
    ]
    cursor.executemany(
        """
        INSERT INTO quotation_items (
            quotation_id, item_no, model_id, description, quantity, unit_price, discount_pct, line_total, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        quotation_items,
    )

    projects = [
        ("PRJ-2025-014", 2, quotation_ids["QUO-2025-014"], "Tofas Press Transfer Upgrade", "commissioning", "Gokhan Tunc", "2025-10-01", "2026-04-25", None, 865000, "EUR", "Site SAT planned for week 18."),
        ("PRJ-2025-021", 3, quotation_ids["QUO-2026-001"], "Arcelik Compact Feed Line", "installation", "Merve Korkmaz", "2026-01-10", "2026-05-30", None, 438000, "EUR", "Mechanical installation ongoing."),
        ("PRJ-2024-011", 1, None, "Ford Otosan CTL Retrofit", "completed", "Gokhan Tunc", "2024-02-01", "2024-11-15", "2024-11-22", 1320000, "EUR", "Delivered with SAT acceptance."),
        ("PRJ-2025-018", 5, None, "Beko Romania Decoiler Renewal", "manufacturing", "Merve Korkmaz", "2025-12-05", "2026-06-12", None, 510000, "EUR", "Double mandrel decoiler line."),
    ]
    cursor.executemany(
        """
        INSERT INTO projects (
            code, customer_id, quotation_id, title, status, project_manager, start_date,
            target_completion, actual_completion, contract_value, currency, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        projects,
    )

    cursor.execute("SELECT id, code FROM projects")
    project_ids = {code: project_id for project_id, code in cursor.fetchall()}

    milestones = [
        (project_ids["PRJ-2025-014"], "Kickoff", "2025-10-03", "2025-10-03", "completed", "Completed on time"),
        (project_ids["PRJ-2025-014"], "FAT", "2026-02-10", "2026-02-12", "completed", "Minor punch list closed"),
        (project_ids["PRJ-2025-014"], "Shipment", "2026-03-20", "2026-03-18", "completed", "Shipped early"),
        (project_ids["PRJ-2025-014"], "SAT", "2026-04-24", None, "in_progress", "Customer maintenance team onsite"),
        (project_ids["PRJ-2025-021"], "GA Approval", "2026-01-25", "2026-01-30", "completed", "Late approval"),
        (project_ids["PRJ-2025-021"], "Mechanical Installation", "2026-04-10", None, "in_progress", "Mechanical assembly at 70%"),
        (project_ids["PRJ-2025-021"], "Electrical Commissioning", "2026-05-15", None, "pending", ""),
        (project_ids["PRJ-2024-011"], "SAT", "2024-11-15", "2024-11-22", "completed", "Final sign-off"),
        (project_ids["PRJ-2025-018"], "Design Freeze", "2026-02-15", "2026-02-10", "completed", "Approved"),
        (project_ids["PRJ-2025-018"], "Main Assembly", "2026-04-30", None, "in_progress", "Mandrel machining completed"),
    ]
    cursor.executemany(
        "INSERT INTO project_milestones (project_id, name, planned_date, actual_date, status, notes) VALUES (?, ?, ?, ?, ?, ?)",
        milestones,
    )

    project_machines = [
        (project_ids["PRJ-2025-014"], model_ids["LD-800"], "SN-LD800-25014", "Loop Drive 800", 1, "commissioned"),
        (project_ids["PRJ-2025-014"], model_ids["PTR-6A"], "SN-PTR6A-25014", "Transfer Robot 6 Axis", 1, "installed"),
        (project_ids["PRJ-2025-021"], model_ids["CD-300"], "SN-CD300-25021", "Compact Drive 300", 1, "installed"),
        (project_ids["PRJ-2025-021"], model_ids["MD-5T"], "SN-MD5T-25021", "Mechanical Decoiler 5T", 1, "installed"),
        (project_ids["PRJ-2024-011"], model_ids["CTL-1600HP"], "SN-CTL1600-24011", "High Performance CTL 1600", 1, "commissioned"),
        (project_ids["PRJ-2025-018"], model_ids["DMD-20T"], "SN-DMD20T-25018", "Double Mandrel Decoiler 20T", 1, "in_production"),
    ]
    cursor.executemany(
        """
        INSERT INTO project_machines (project_id, model_id, serial_number, description, quantity, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        project_machines,
    )

    line_configurations = [
        ("LCFG-2026-001", project_ids["PRJ-2025-021"], 2, "Arcelik Side Panel Compact Line", "compact_feed_line", "galvanized_steel", 180, 320, 0.7, 1.6, 28, 4500, 250, "Compact line for appliance side panel die set."),
        ("LCFG-2026-002", project_ids["PRJ-2025-014"], 1, "Tofas Transfer Upgrade Line", "transfer_line", "mild_steel", 450, 800, 1.0, 3.2, 55, 9000, 630, "Loop drive plus transfer robot."),
        ("LCFG-2025-003", project_ids["PRJ-2024-011"], None, "Ford CTL Retrofit", "ctl_line", "cold_rolled_steel", 800, 1600, 0.8, 4.5, 65, 18000, None, "Dynamic shear CTL line."),
    ]
    cursor.executemany(
        """
        INSERT INTO line_configurations (
            code, project_id, opportunity_id, name, line_type, material_type,
            strip_width_min_mm, strip_width_max_mm, strip_thickness_min_mm,
            strip_thickness_max_mm, line_speed_mpm, coil_weight_max_kg,
            press_force_ton, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        line_configurations,
    )

    cursor.execute("SELECT id, code FROM line_configurations")
    config_ids = {code: config_id for config_id, code in cursor.fetchall()}

    bom_components = [
        (config_ids["LCFG-2026-001"], model_ids["MD-5T"], "Entry Decoiler", "decoiler", 1, "Mechanical decoiler"),
        (config_ids["LCFG-2026-001"], model_ids["CD-300"], "Servo Feeder", "feeder", 1, "Compact drive"),
        (config_ids["LCFG-2026-001"], model_ids["TG-500"], "Triggered Sync Module", "control", 1, "Press sync"),
        (config_ids["LCFG-2026-002"], model_ids["LD-800"], "Loop Feed Line", "feeder", 1, "Main feed line"),
        (config_ids["LCFG-2026-002"], model_ids["PTR-6A"], "Transfer Robot", "transfer", 1, "Robot transfer"),
        (config_ids["LCFG-2025-003"], model_ids["DMD-20T"], "Double Mandrel Decoiler", "decoiler", 1, "Entry line"),
        (config_ids["LCFG-2025-003"], model_ids["CTL-1600HP"], "CTL Core Module", "ctl", 1, "Shear and stacker"),
    ]
    cursor.executemany(
        """
        INSERT INTO bom_components (line_config_id, model_id, component_name, component_type, quantity, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        bom_components,
    )

    installed_base = [
        ("IB-0001", 2, project_ids["PRJ-2025-014"], model_ids["LD-800"], "SN-LD800-25014", "Tofas Loop Feed Line", "2026-03-30", "2027-03-30", "Bursa Press Shop", "Line 5", "operational", "SAT in progress, production trials started."),
        ("IB-0002", 2, project_ids["PRJ-2025-014"], model_ids["PTR-6A"], "SN-PTR6A-25014", "Tofas Transfer Robot Cell", "2026-03-31", "2027-03-31", "Bursa Press Shop", "Line 5", "operational", "Transfer robot paired with loop drive."),
        ("IB-0003", 3, project_ids["PRJ-2025-021"], model_ids["CD-300"], "SN-CD300-25021", "Arcelik Compact Feed Line", "2026-04-08", "2027-04-08", "Eskisehir Press Shop", "Appliance Line 2", "maintenance", "Commissioning paused for encoder drift."),
        ("IB-0004", 1, project_ids["PRJ-2024-011"], model_ids["CTL-1600HP"], "SN-CTL1600-24011", "Ford CTL 1600HP", "2024-11-22", "2025-11-22", "Golcuk Service Center", "Coil Prep Line", "operational", "Production reference line."),
        ("IB-0005", 5, project_ids["PRJ-2025-018"], model_ids["DMD-20T"], "SN-DMD20T-25018", "Beko Double Mandrel Decoiler", "2025-12-18", "2026-12-18", "Gaesti Press Shop", "Entry Line", "operational", "High uptime coil change line."),
    ]
    cursor.executemany(
        """
        INSERT INTO installed_base (
            code, customer_id, project_id, model_id, serial_number, machine_name,
            installation_date, warranty_expiry, location_plant, location_line, status, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        installed_base,
    )

    cursor.execute("SELECT id, code FROM installed_base")
    machine_ids = {code: machine_id for machine_id, code in cursor.fetchall()}

    commissioning_reports = [
        (machine_ids["IB-0001"], project_ids["PRJ-2025-014"], "2026-04-12", "Emrah Uslu", "conditional", 0.12, 51.5, 4.6, 73.0, 1, 0, '["Encoder drift review", "Customer recipe approval"]', "Line producing trial parts."),
        (machine_ids["IB-0003"], project_ids["PRJ-2025-021"], "2026-04-10", "Emrah Uslu", "failed", 0.28, 24.0, 5.1, 71.0, 1, 0, '["Encoder recalibration", "Feed roll pressure adjustment"]', "Feed accuracy outside spec."),
        (machine_ids["IB-0004"], project_ids["PRJ-2024-011"], "2024-11-22", "Emrah Uslu", "passed", 0.09, 68.0, 3.8, 74.0, 1, 1, '[]', "Customer signed off."),
    ]
    cursor.executemany(
        """
        INSERT INTO commissioning_reports (
            installed_base_id, project_id, report_date, engineer, status, feed_accuracy_mm,
            speed_achieved_mpm, vibration_level, noise_level_db, safety_check_passed,
            customer_signoff, punch_list, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        commissioning_reports,
    )

    service_tickets = [
        ("SRV-2026-001", 3, machine_ids["IB-0003"], "Encoder drift after coil change", "Feed length deviation exceeded target after coil change.", "breakdown", "high", "open", "Cem Toker", "Onur Demir", "2026-04-11", None, None, 0.0, 3.5, 420.0),
        ("SRV-2026-002", 2, machine_ids["IB-0002"], "Vacuum gripper pressure fluctuation", "Transfer robot vacuum status intermittently drops.", "predictive", "medium", "assigned", "Cem Toker", "Selin Acar", "2026-04-13", None, None, 0.0, 1.5, 180.0),
        ("SRV-2026-003", 1, machine_ids["IB-0004"], "Quarterly CTL preventive maintenance", "Scheduled PM for CTL line.", "preventive", "low", "in_progress", "Bora Ural", "Mert Yilmaz", "2026-04-15", None, None, 0.0, 5.0, 320.0),
        ("SRV-2025-028", 5, machine_ids["IB-0005"], "Mandrel seal replacement", "Hydraulic seal leakage during coil change.", "warranty", "medium", "resolved", "Bora Ural", "Ana Popescu", "2026-03-10", "2026-03-12", "Seal kit replaced and pressure tested.", 8.0, 6.0, 980.0),
    ]
    cursor.executemany(
        """
        INSERT INTO service_tickets (
            code, customer_id, installed_base_id, title, description, type, priority, status,
            assigned_to, reported_by, reported_date, resolved_date, resolution, travel_hours, work_hours, cost
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        service_tickets,
    )

    cursor.execute("SELECT id, code FROM service_tickets")
    ticket_ids = {code: ticket_id for ticket_id, code in cursor.fetchall()}

    maintenance_plans = [
        (machine_ids["IB-0001"], "Servo feed calibration", "months", 3, "2026-01-10", "2026-04-20", "Cem Toker", "Check encoder scaling, feed roll pressure and pilot release."),
        (machine_ids["IB-0002"], "Vacuum cup inspection", "weeks", 4, "2026-03-15", "2026-04-15", "Cem Toker", "Inspect vacuum lines and cups."),
        (machine_ids["IB-0004"], "CTL shear inspection", "hours", 750, "2026-02-05", "2026-05-01", "Bora Ural", "Inspect shear blade and encoder."),
        (machine_ids["IB-0005"], "Mandrel lubrication", "weeks", 8, "2026-02-25", "2026-04-22", "Bora Ural", "Lubricate mandrel bearings and hydraulic head."),
    ]
    cursor.executemany(
        """
        INSERT INTO maintenance_plans (
            installed_base_id, task_name, interval_type, interval_value, last_performed, next_due, assigned_to, instructions
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        maintenance_plans,
    )

    cursor.execute("SELECT id, installed_base_id, task_name FROM maintenance_plans")
    plan_ids = {(installed_base_id, task_name): plan_id for plan_id, installed_base_id, task_name in cursor.fetchall()}

    maintenance_logs = [
        (plan_ids[(machine_ids["IB-0001"], "Servo feed calibration")], machine_ids["IB-0001"], None, "2026-01-10", "Cem Toker", "preventive", "Servo feed calibration completed.", '["ENC-CTL-01"]', 2.5, 210.0, "Baseline calibration stored."),
        (plan_ids[(machine_ids["IB-0002"], "Vacuum cup inspection")], machine_ids["IB-0002"], ticket_ids["SRV-2026-002"], "2026-03-15", "Cem Toker", "predictive", "Vacuum cup wear inspection.", '["VAC-24V-06"]', 1.0, 95.0, "Cup wear at 60%."),
        (plan_ids[(machine_ids["IB-0004"], "CTL shear inspection")], machine_ids["IB-0004"], ticket_ids["SRV-2026-003"], "2026-02-05", "Bora Ural", "preventive", "Shear and encoder inspection.", '["BRG-6208-2RS"]', 4.5, 280.0, "Encoder clean, bearings acceptable."),
        (plan_ids[(machine_ids["IB-0005"], "Mandrel lubrication")], machine_ids["IB-0005"], ticket_ids["SRV-2025-028"], "2026-03-12", "Bora Ural", "corrective", "Seal replacement and lubrication.", '["SEAL-DMD-20T"]', 6.0, 980.0, "Leak resolved."),
    ]
    cursor.executemany(
        """
        INSERT INTO maintenance_logs (
            plan_id, installed_base_id, service_ticket_id, performed_date, performed_by,
            type, description, parts_used, duration_hours, cost, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        maintenance_logs,
    )

    spare_parts = [
        ("SM-400", "Servo Motor 4.0 kW", "servo_motor", '["CD-300","TG-500"]', "pcs", 1850, 28, "Yaskawa", 1, "Line-stopping item for compact drives."),
        ("ENC-CTL-01", "Feed Encoder Kit", "encoder", '["CD-300","LD-800","CTL-1600HP"]', "pcs", 320, 14, "SICK", 1, "Used for feed calibration."),
        ("BRG-6208-2RS", "Straightener Bearing Set", "bearing", '["LD-800","CTL-1600HP"]', "set", 140, 7, "SKF", 1, "Roller bearing set."),
        ("VAC-24V-06", "Vacuum Valve Kit 24V", "pneumatic", '["PTR-6A","DSK-800"]', "pcs", 95, 10, "SMC", 0, "Vacuum valve replacement kit."),
        ("SEAL-DMD-20T", "Mandrel Seal Kit 20T", "seal", '["DMD-20T"]', "kit", 260, 21, "Parker", 1, "Hydraulic mandrel seal kit."),
        ("BELT-SF-800", "Servo Feed Timing Belt", "belt", '["LD-800"]', "pcs", 180, 12, "Optibelt", 0, "Main drive timing belt."),
    ]
    cursor.executemany(
        """
        INSERT INTO spare_parts (
            part_number, name, category, applicable_models, unit, unit_cost, lead_time_days, supplier, is_critical, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        spare_parts,
    )

    cursor.execute("SELECT id, part_number FROM spare_parts")
    part_ids = {part_number: part_id for part_id, part_number in cursor.fetchall()}

    stock = [
        (part_ids["SM-400"], "main_warehouse", 1, 2, 4, "2026-03-01"),
        (part_ids["ENC-CTL-01"], "main_warehouse", 6, 3, 10, "2026-04-01"),
        (part_ids["BRG-6208-2RS"], "main_warehouse", 2, 4, 10, "2026-03-25"),
        (part_ids["VAC-24V-06"], "main_warehouse", 8, 4, 12, "2026-04-08"),
        (part_ids["SEAL-DMD-20T"], "main_warehouse", 1, 1, 3, "2026-03-12"),
        (part_ids["BELT-SF-800"], "service_van_1", 0, 1, 2, "2026-02-18"),
    ]
    cursor.executemany(
        """
        INSERT INTO spare_part_stock (part_id, location, quantity, min_stock, max_stock, last_restock_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        stock,
    )

    technical_documents = [
        ("Compact Drive Selection Guide", "datasheet", model_ids["CD-300"], family_ids["PF-CD"], "en", "data/demo_docs/compact_drive_selection_guide.txt", "products_compact_drives", "2026.1", '{"demo": true}'),
        ("Servo Drive Alarm Matrix", "service_bulletin", model_ids["LD-800"], family_ids["PF-LD"], "en", "data/demo_docs/servo_drive_alarm_matrix.txt", "service_maintenance", "2026.1", '{"demo": true}'),
        ("CTL Line Commissioning Checklist", "commissioning_checklist", model_ids["CTL-1600HP"], family_ids["PF-HPCTL"], "en", "data/demo_docs/ctl_commissioning_checklist.txt", "installation_commissioning", "2026.1", '{"demo": true}'),
        ("Critical Spare Parts Stocking Guide", "manual", None, None, "en", "data/demo_docs/spare_parts_critical_stock.txt", "spare_parts", "2026.1", '{"demo": true}'),
        ("Commercial Terms 2026", "manual", None, None, "en", "data/demo_docs/quotation_terms_2026.txt", "sales_and_quotations", "2026.1", '{"demo": true}'),
        ("CE Compliance for Press Feeding Lines", "certificate", None, None, "en", "data/demo_docs/ce_compliance_press_lines.txt", "quality_documents", "2026.1", '{"demo": true}'),
        ("Destacker and Blank Line Integration Concept", "drawing", None, family_ids["PF-SP"], "en", "data/demo_docs/special_project_destacker_layout.txt", "special_projects", "2026.1", '{"demo": true}'),
    ]
    cursor.executemany(
        """
        INSERT INTO technical_documents (
            title, doc_type, model_id, family_id, language, file_path, rag_collection, version, metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        technical_documents,
    )

    sensor_streams = [
        (machine_ids["IB-0001"], "2026-04-16 08:10:00", 72.0, 299.96, 300.00, 52.0, 640.0, 58.0, 3.8, 5.9, 1, 125430, 1, 0.03, 8.2),
        (machine_ids["IB-0002"], "2026-04-16 08:11:00", 81.0, None, None, 16.0, None, 61.0, 5.2, 4.8, 2, 45210, 1, 0.00, 7.0),
        (machine_ids["IB-0003"], "2026-04-16 08:12:00", 89.0, 300.28, 300.00, 24.0, 420.0, 67.0, 6.5, 5.1, 1, 21450, 1, 0.24, 8.8),
        (machine_ids["IB-0004"], "2026-04-16 08:13:00", 63.0, 1500.05, 1500.00, 67.0, 980.0, 55.0, 3.3, 6.2, 1, 989120, 1, 0.02, 21.5),
        (machine_ids["IB-0005"], "2026-04-16 08:14:00", 76.0, None, None, 30.0, 710.0, 73.0, 8.2, 5.6, 1, 187220, 0, 0.01, 12.0),
    ]
    cursor.executemany(
        """
        INSERT INTO sensor_streams (
            machine_id, timestamp, servo_load_pct, feed_length_actual_mm, feed_length_target_mm,
            line_speed_mpm, coil_remaining_diameter_mm, bearing_temperature_c, vibration_mm_s,
            air_pressure_bar, vacuum_status, cycle_count, lubrication_ok, encoder_drift_mm, power_consumption_kw
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sensor_streams,
    )

    cursor.execute("SELECT id, machine_id FROM sensor_streams")
    stream_ids = {}
    for stream_id, machine_id in cursor.fetchall():
        stream_ids[machine_id] = stream_id

    alerts = [
        (machine_ids["IB-0003"], stream_ids[machine_ids["IB-0003"]], "2026-04-16 08:12:05", "servo_overload", "critical", 89.0, 85.0, "Servo load above threshold on Arcelik compact line.", 0, None, 0, None),
        (machine_ids["IB-0003"], stream_ids[machine_ids["IB-0003"]], "2026-04-16 08:12:10", "encoder_drift", "critical", 0.24, 0.10, "Encoder drift exceeds acceptable limit.", 0, None, 0, None),
        (machine_ids["IB-0002"], stream_ids[machine_ids["IB-0002"]], "2026-04-16 08:11:05", "low_air_pressure", "warning", 4.8, 4.5, "Vacuum and air pressure unstable on transfer robot cell.", 0, None, 0, None),
        (machine_ids["IB-0005"], stream_ids[machine_ids["IB-0005"]], "2026-04-16 08:14:15", "high_vibration", "warning", 8.2, 7.5, "Double mandrel decoiler vibration elevated.", 0, None, 0, None),
    ]
    cursor.executemany(
        """
        INSERT INTO machine_alerts (
            machine_id, sensor_stream_id, timestamp, alert_type, severity, value, threshold,
            message, acknowledged, acknowledged_by, resolved, resolved_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        alerts,
    )

    line_runs = [
        (machine_ids["IB-0001"], "2026-04-15", "morning", 18250, 19000, 17890, 360, 42, 48.0, 3, "SAT production trial"),
        (machine_ids["IB-0003"], "2026-04-15", "morning", 6250, 9000, 5980, 270, 95, 22.0, 1, "Commissioning trial with feed deviation issues"),
        (machine_ids["IB-0004"], "2026-04-15", "afternoon", 14500, 15000, 14470, 30, 18, 64.0, 4, "Stable CTL production"),
        (machine_ids["IB-0005"], "2026-04-15", "night", 9800, 10500, 9720, 80, 24, 29.0, 2, "Coil change after midnight"),
    ]
    cursor.executemany(
        """
        INSERT INTO line_runs (
            machine_id, run_date, shift, total_strokes, target_strokes, good_parts, reject_parts,
            downtime_minutes, avg_speed_spm, coils_processed, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        line_runs,
    )

    quality_events = [
        ("QE-2026-001", "installed_base", machine_ids["IB-0003"], "acceptance_test", "major", "Feed accuracy out of spec during SAT", "Arcelik compact line SAT feed accuracy exceeded limit.", "Encoder drift after coil change", "Recalibrate encoder and verify feed roll pressure.", "investigating", "Ece Sahin", "2026-04-11", None),
        ("QE-2026-002", "project", project_ids["PRJ-2025-014"], "ce_review", "minor", "CE dossier final review", "Final review of CE technical file before customer handover.", None, "Complete residual risk annex.", "open", "Ece Sahin", "2026-04-09", None),
        ("QE-2026-003", "service_ticket", ticket_ids["SRV-2025-028"], "warranty_claim", "minor", "Mandrel seal warranty claim", "Seal kit failure within warranty window.", "Seal wear due to pressure spike", "Pressure setting updated and seals replaced.", "resolved", "Ece Sahin", "2026-03-11", "2026-03-20"),
    ]
    cursor.executemany(
        """
        INSERT INTO quality_events (
            code, related_entity_type, related_entity_id, event_type, severity, title,
            description, root_cause, corrective_action, status, assigned_to, created_at, closed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        quality_events,
    )

    conn.commit()


async def seed_rag_documents(db: SQLiteManager) -> None:
    rag = RAGPipeline()
    rag.vector_store.clear_all()

    docs_dir = Path("data/demo_docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    for document in DEMO_DOCUMENTS:
        file_path = docs_dir / document["file_name"]
        file_path.write_text(document["content"].strip(), encoding="utf-8")

        await rag.ingest_async(
            file_path=str(file_path),
            department=document["collection"],
            metadata={
                "title": document["title"],
                "doc_type": document["doc_type"],
                "language": "en",
                "source": str(file_path),
                "demo": True,
            },
        )

        await db.register_uploaded_document(
            original_filename=document["file_name"],
            stored_filename=document["file_name"],
            file_path=str(file_path),
            mime_type="text/plain",
            file_size_bytes=len(document["content"].encode("utf-8")),
            document_role="general_document",
            rag_collection=document["collection"],
            metadata={"demo": True, "title": document["title"]},
        )


def print_summary(db: SQLiteManager) -> None:
    tables = [
        "customers",
        "product_families",
        "product_models",
        "project_opportunities",
        "quotations",
        "projects",
        "installed_base",
        "service_tickets",
        "spare_parts",
        "sensor_streams",
        "machine_alerts",
        "technical_documents",
        "uploaded_documents",
    ]
    print("Seed summary:")
    for table in tables:
        result = db.execute_query(f"SELECT COUNT(*) AS count FROM {table}", fetch_one=True)
        print(f"{table:22} {result['count']}")


async def main() -> None:
    db = SQLiteManager()
    db.initialize_schema()
    reset_database(db)
    seed_relational_data(db)
    await seed_rag_documents(db)
    print_summary(db)
    await db.async_close()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
