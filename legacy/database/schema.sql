-- =============================================================================
-- Coiltech Industrial Automation Platform — Database Schema
-- Domain: Press feeding systems, decoilers, CTL lines, transfer systems
-- =============================================================================

PRAGMA foreign_keys = ON;

-- =============================================================================
-- CUSTOMERS & CONTACTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- e.g. "CUST-001"
    name TEXT NOT NULL,                       -- company name
    country TEXT NOT NULL DEFAULT 'TR',
    city TEXT,
    industry TEXT,                            -- automotive, appliance, general stamping
    tier TEXT CHECK(tier IN ('strategic','key','standard','prospect')) DEFAULT 'standard',
    website TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_customers_code ON customers(code);
CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country);

CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    name TEXT NOT NULL,
    title TEXT,
    email TEXT,
    phone TEXT,
    department TEXT,                          -- engineering, purchasing, maintenance
    is_primary INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_contacts_customer ON contacts(customer_id);

-- =============================================================================
-- PRODUCT CATALOG
-- =============================================================================
CREATE TABLE IF NOT EXISTS product_families (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- e.g. "PF-CD" for Compact Drives
    name TEXT NOT NULL,
    category TEXT NOT NULL,                   -- press_feeding, decoiler, ctl_line, transfer, special
    description TEXT,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pf_category ON product_families(category);

CREATE TABLE IF NOT EXISTS product_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id INTEGER NOT NULL REFERENCES product_families(id),
    model_code TEXT NOT NULL UNIQUE,          -- e.g. "CD-300"
    name TEXT NOT NULL,
    max_strip_width_mm REAL,
    max_strip_thickness_mm REAL,
    max_feed_length_mm REAL,
    max_speed_mpm REAL,                      -- meters per minute
    max_coil_weight_kg REAL,
    max_coil_od_mm REAL,
    press_force_ton REAL,
    servo_power_kw REAL,
    weight_kg REAL,
    description TEXT,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pm_family ON product_models(family_id);
CREATE INDEX IF NOT EXISTS idx_pm_model_code ON product_models(model_code);

CREATE TABLE IF NOT EXISTS machine_capabilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES product_models(id),
    capability_name TEXT NOT NULL,            -- "zigzag_mode", "pilot_release", "loop_control"
    capability_value TEXT,
    unit TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_mc_model ON machine_capabilities(model_id);

-- =============================================================================
-- SALES PIPELINE
-- =============================================================================
CREATE TABLE IF NOT EXISTS project_opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "OPP-2024-001"
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    title TEXT NOT NULL,
    description TEXT,
    stage TEXT NOT NULL CHECK(stage IN (
        'lead','qualification','proposal','negotiation','won','lost','cancelled'
    )) DEFAULT 'lead',
    estimated_value REAL,
    currency TEXT DEFAULT 'EUR',
    probability_pct INTEGER DEFAULT 10,
    expected_close_date DATE,
    owner TEXT,                               -- sales engineer
    source TEXT,                              -- exhibition, referral, website, cold_call
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_opp_customer ON project_opportunities(customer_id);
CREATE INDEX IF NOT EXISTS idx_opp_stage ON project_opportunities(stage);

CREATE TABLE IF NOT EXISTS quotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "QUO-2024-001"
    opportunity_id INTEGER REFERENCES project_opportunities(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    revision INTEGER DEFAULT 1,
    status TEXT NOT NULL CHECK(status IN (
        'draft','sent','under_review','accepted','rejected','expired'
    )) DEFAULT 'draft',
    total_amount REAL,
    currency TEXT DEFAULT 'EUR',
    validity_days INTEGER DEFAULT 30,
    delivery_weeks INTEGER,
    payment_terms TEXT,
    incoterm TEXT DEFAULT 'EXW',
    notes TEXT,
    prepared_by TEXT,
    issued_date DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_quo_customer ON quotations(customer_id);
CREATE INDEX IF NOT EXISTS idx_quo_opportunity ON quotations(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_quo_status ON quotations(status);

CREATE TABLE IF NOT EXISTS quotation_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quotation_id INTEGER NOT NULL REFERENCES quotations(id),
    item_no INTEGER NOT NULL,
    model_id INTEGER REFERENCES product_models(id),
    description TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    unit_price REAL,
    discount_pct REAL DEFAULT 0,
    line_total REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_qi_quotation ON quotation_items(quotation_id);

-- =============================================================================
-- PROJECT EXECUTION
-- =============================================================================
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "PRJ-2024-001"
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    quotation_id INTEGER REFERENCES quotations(id),
    title TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'planning','engineering','manufacturing','testing',
        'shipping','installation','commissioning','completed','on_hold','cancelled'
    )) DEFAULT 'planning',
    project_manager TEXT,
    start_date DATE,
    target_completion DATE,
    actual_completion DATE,
    contract_value REAL,
    currency TEXT DEFAULT 'EUR',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prj_customer ON projects(customer_id);
CREATE INDEX IF NOT EXISTS idx_prj_status ON projects(status);

CREATE TABLE IF NOT EXISTS project_milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name TEXT NOT NULL,
    planned_date DATE,
    actual_date DATE,
    status TEXT CHECK(status IN ('pending','in_progress','completed','delayed','skipped'))
        DEFAULT 'pending',
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_pm_project ON project_milestones(project_id);

CREATE TABLE IF NOT EXISTS project_machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id),
    model_id INTEGER REFERENCES product_models(id),
    serial_number TEXT,
    description TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    status TEXT CHECK(status IN (
        'ordered','in_production','testing','shipped','installed','commissioned'
    )) DEFAULT 'ordered'
);
CREATE INDEX IF NOT EXISTS idx_pmach_project ON project_machines(project_id);

-- =============================================================================
-- LINE CONFIGURATIONS
-- =============================================================================
CREATE TABLE IF NOT EXISTS line_configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "LCFG-001"
    project_id INTEGER REFERENCES projects(id),
    opportunity_id INTEGER REFERENCES project_opportunities(id),
    name TEXT NOT NULL,
    line_type TEXT,                           -- "compact_feed_line", "ctl_line", "blank_line"
    material_type TEXT,                       -- steel, aluminum, stainless, copper
    strip_width_min_mm REAL,
    strip_width_max_mm REAL,
    strip_thickness_min_mm REAL,
    strip_thickness_max_mm REAL,
    line_speed_mpm REAL,
    coil_weight_max_kg REAL,
    press_force_ton REAL,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_lcfg_project ON line_configurations(project_id);

CREATE TABLE IF NOT EXISTS bom_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_config_id INTEGER REFERENCES line_configurations(id),
    model_id INTEGER REFERENCES product_models(id),
    component_name TEXT NOT NULL,
    component_type TEXT,                      -- decoiler, straightener, feeder, transfer, conveyor
    quantity INTEGER DEFAULT 1,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_bom_config ON bom_components(line_config_id);

-- =============================================================================
-- INSTALLED BASE
-- =============================================================================
CREATE TABLE IF NOT EXISTS installed_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "IB-0001"
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    project_id INTEGER REFERENCES projects(id),
    model_id INTEGER REFERENCES product_models(id),
    serial_number TEXT,
    machine_name TEXT NOT NULL,
    installation_date DATE,
    warranty_expiry DATE,
    location_plant TEXT,
    location_line TEXT,
    status TEXT CHECK(status IN ('operational','maintenance','offline','decommissioned'))
        DEFAULT 'operational',
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ib_customer ON installed_base(customer_id);
CREATE INDEX IF NOT EXISTS idx_ib_status ON installed_base(status);
CREATE INDEX IF NOT EXISTS idx_ib_serial ON installed_base(serial_number);

-- =============================================================================
-- COMMISSIONING
-- =============================================================================
CREATE TABLE IF NOT EXISTS commissioning_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    installed_base_id INTEGER NOT NULL REFERENCES installed_base(id),
    project_id INTEGER REFERENCES projects(id),
    report_date DATE NOT NULL,
    engineer TEXT NOT NULL,
    status TEXT CHECK(status IN ('pending','in_progress','passed','failed','conditional'))
        DEFAULT 'pending',
    feed_accuracy_mm REAL,
    speed_achieved_mpm REAL,
    vibration_level REAL,
    noise_level_db REAL,
    safety_check_passed INTEGER DEFAULT 0,
    customer_signoff INTEGER DEFAULT 0,
    punch_list TEXT,                          -- JSON array of remaining items
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cr_ib ON commissioning_reports(installed_base_id);

-- =============================================================================
-- SERVICE & MAINTENANCE
-- =============================================================================
CREATE TABLE IF NOT EXISTS service_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "SRV-2024-001"
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    installed_base_id INTEGER REFERENCES installed_base(id),
    title TEXT NOT NULL,
    description TEXT,
    type TEXT CHECK(type IN ('breakdown','preventive','predictive','warranty','upgrade'))
        NOT NULL DEFAULT 'breakdown',
    priority TEXT CHECK(priority IN ('low','medium','high','critical')) DEFAULT 'medium',
    status TEXT CHECK(status IN ('open','assigned','in_progress','waiting_parts',
        'resolved','closed','cancelled')) DEFAULT 'open',
    assigned_to TEXT,
    reported_by TEXT,
    reported_date DATE DEFAULT (DATE('now')),
    resolved_date DATE,
    resolution TEXT,
    travel_hours REAL,
    work_hours REAL,
    cost REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_srv_customer ON service_tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_srv_ib ON service_tickets(installed_base_id);
CREATE INDEX IF NOT EXISTS idx_srv_status ON service_tickets(status);

CREATE TABLE IF NOT EXISTS maintenance_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    installed_base_id INTEGER NOT NULL REFERENCES installed_base(id),
    task_name TEXT NOT NULL,
    interval_type TEXT CHECK(interval_type IN ('hours','days','weeks','months','cycles'))
        NOT NULL,
    interval_value INTEGER NOT NULL,
    last_performed DATE,
    next_due DATE,
    assigned_to TEXT,
    instructions TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_mp_ib ON maintenance_plans(installed_base_id);
CREATE INDEX IF NOT EXISTS idx_mp_next_due ON maintenance_plans(next_due);

CREATE TABLE IF NOT EXISTS maintenance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER REFERENCES maintenance_plans(id),
    installed_base_id INTEGER NOT NULL REFERENCES installed_base(id),
    service_ticket_id INTEGER REFERENCES service_tickets(id),
    performed_date DATE NOT NULL,
    performed_by TEXT NOT NULL,
    type TEXT CHECK(type IN ('preventive','corrective','predictive','emergency'))
        NOT NULL,
    description TEXT,
    parts_used TEXT,                          -- JSON array
    duration_hours REAL,
    cost REAL,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ml_ib ON maintenance_logs(installed_base_id);
CREATE INDEX IF NOT EXISTS idx_ml_date ON maintenance_logs(performed_date);

-- =============================================================================
-- SPARE PARTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS spare_parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_number TEXT NOT NULL UNIQUE,         -- "SP-SRV-001"
    name TEXT NOT NULL,
    category TEXT,                            -- servo_motor, bearing, belt, seal, electronic, pneumatic
    applicable_models TEXT,                   -- JSON array of model_codes
    unit TEXT DEFAULT 'pcs',
    unit_cost REAL,
    lead_time_days INTEGER,
    supplier TEXT,
    is_critical INTEGER DEFAULT 0,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sp_part_number ON spare_parts(part_number);
CREATE INDEX IF NOT EXISTS idx_sp_category ON spare_parts(category);

CREATE TABLE IF NOT EXISTS spare_part_stock (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id INTEGER NOT NULL REFERENCES spare_parts(id),
    location TEXT DEFAULT 'main_warehouse',
    quantity INTEGER NOT NULL DEFAULT 0,
    min_stock INTEGER DEFAULT 0,
    max_stock INTEGER,
    last_restock_date DATE,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sps_part ON spare_part_stock(part_id);

-- =============================================================================
-- TECHNICAL DOCUMENTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS technical_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    doc_type TEXT CHECK(doc_type IN (
        'manual','datasheet','drawing','certificate','sop',
        'commissioning_checklist','service_bulletin','training'
    )) NOT NULL,
    model_id INTEGER REFERENCES product_models(id),
    family_id INTEGER REFERENCES product_families(id),
    language TEXT DEFAULT 'en',
    file_path TEXT,
    rag_collection TEXT,                     -- which chroma collection it was ingested into
    version TEXT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                             -- JSON blob
);
CREATE INDEX IF NOT EXISTS idx_td_model ON technical_documents(model_id);
CREATE INDEX IF NOT EXISTS idx_td_type ON technical_documents(doc_type);

CREATE TABLE IF NOT EXISTS uploaded_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    stored_filename TEXT,
    file_path TEXT NOT NULL,
    mime_type TEXT,
    file_size_bytes INTEGER,
    document_role TEXT CHECK(document_role IN (
        'product_document','quotation_attachment','project_document','service_document',
        'quality_document','commissioning_document','general_document'
    )) DEFAULT 'general_document',
    customer_id INTEGER REFERENCES customers(id),
    project_id INTEGER REFERENCES projects(id),
    installed_base_id INTEGER REFERENCES installed_base(id),
    technical_document_id INTEGER REFERENCES technical_documents(id),
    rag_collection TEXT,
    upload_source TEXT DEFAULT 'api',
    uploaded_by TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ud_customer ON uploaded_documents(customer_id);
CREATE INDEX IF NOT EXISTS idx_ud_project ON uploaded_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_ud_rag_collection ON uploaded_documents(rag_collection);

-- =============================================================================
-- DIGITAL TWIN — SENSOR DATA & ALERTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS sensor_streams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES installed_base(id),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    servo_load_pct REAL,
    feed_length_actual_mm REAL,
    feed_length_target_mm REAL,
    line_speed_mpm REAL,
    coil_remaining_diameter_mm REAL,
    bearing_temperature_c REAL,
    vibration_mm_s REAL,
    air_pressure_bar REAL,
    vacuum_status INTEGER,                   -- 0=off, 1=ok, 2=low
    cycle_count INTEGER,
    lubrication_ok INTEGER,                  -- 0=overdue, 1=ok
    encoder_drift_mm REAL,
    power_consumption_kw REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ss_machine ON sensor_streams(machine_id);
CREATE INDEX IF NOT EXISTS idx_ss_timestamp ON sensor_streams(timestamp);

CREATE TABLE IF NOT EXISTS machine_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES installed_base(id),
    sensor_stream_id INTEGER REFERENCES sensor_streams(id),
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT NOT NULL,                -- servo_overload, feed_deviation, high_vibration, etc.
    severity TEXT CHECK(severity IN ('info','warning','critical','emergency')) NOT NULL,
    value REAL,
    threshold REAL,
    message TEXT,
    acknowledged INTEGER DEFAULT 0,
    acknowledged_by TEXT,
    resolved INTEGER DEFAULT 0,
    resolved_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ma_machine ON machine_alerts(machine_id);
CREATE INDEX IF NOT EXISTS idx_ma_severity ON machine_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_ma_resolved ON machine_alerts(resolved);

-- =============================================================================
-- PRODUCTION / LINE RUNS  (data from customer machines)
-- =============================================================================
CREATE TABLE IF NOT EXISTS line_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES installed_base(id),
    run_date DATE NOT NULL,
    shift TEXT CHECK(shift IN ('morning','afternoon','night')),
    total_strokes INTEGER DEFAULT 0,
    target_strokes INTEGER DEFAULT 0,
    good_parts INTEGER DEFAULT 0,
    reject_parts INTEGER DEFAULT 0,
    downtime_minutes INTEGER DEFAULT 0,
    avg_speed_spm REAL,                      -- strokes per minute
    coils_processed INTEGER DEFAULT 0,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_lr_machine ON line_runs(machine_id);
CREATE INDEX IF NOT EXISTS idx_lr_date ON line_runs(run_date);

-- =============================================================================
-- QUALITY EVENTS
-- =============================================================================
CREATE TABLE IF NOT EXISTS quality_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,                -- "QE-2024-001"
    related_entity_type TEXT,                 -- project, installed_base, service_ticket
    related_entity_id INTEGER,
    event_type TEXT CHECK(event_type IN (
        'ncr','customer_complaint','internal_audit','supplier_issue',
        'ce_review','acceptance_test','warranty_claim'
    )) NOT NULL,
    severity TEXT CHECK(severity IN ('minor','major','critical')) DEFAULT 'minor',
    title TEXT NOT NULL,
    description TEXT,
    root_cause TEXT,
    corrective_action TEXT,
    status TEXT CHECK(status IN ('open','investigating','resolved','closed'))
        DEFAULT 'open',
    assigned_to TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    closed_at DATETIME
);
CREATE INDEX IF NOT EXISTS idx_qe_status ON quality_events(status);
CREATE INDEX IF NOT EXISTS idx_qe_type ON quality_events(event_type);

-- =============================================================================
-- VIEWS
-- =============================================================================

DROP VIEW IF EXISTS v_pipeline_summary;
CREATE VIEW IF NOT EXISTS v_pipeline_summary AS
SELECT
    po.stage,
    COUNT(*) AS opportunity_count,
    SUM(po.estimated_value) AS total_value,
    AVG(po.probability_pct) AS avg_probability
FROM project_opportunities po
WHERE po.stage NOT IN ('lost','cancelled')
GROUP BY po.stage;

DROP VIEW IF EXISTS v_project_status;
CREATE VIEW IF NOT EXISTS v_project_status AS
SELECT
    p.id,
    p.customer_id,
    p.code, p.title, p.status,
    c.name AS customer_name,
    p.project_manager,
    p.contract_value, p.currency, p.target_completion,
    COUNT(pm2.id) AS total_milestones,
    SUM(CASE WHEN pm2.status = 'completed' THEN 1 ELSE 0 END) AS completed_milestones,
    CASE
        WHEN COUNT(pm2.id) = 0 THEN 0
        ELSE ROUND(
            (SUM(CASE WHEN pm2.status = 'completed' THEN 1 ELSE 0 END) * 100.0) / COUNT(pm2.id),
            1
        )
    END AS completion_pct
FROM projects p
JOIN customers c ON p.customer_id = c.id
LEFT JOIN project_milestones pm2 ON p.id = pm2.project_id
GROUP BY p.id;

DROP VIEW IF EXISTS v_open_service_tickets;
CREATE VIEW IF NOT EXISTS v_open_service_tickets AS
SELECT
    st.id,
    st.customer_id,
    st.installed_base_id,
    st.code, st.title, st.type, st.priority, st.status,
    c.name AS customer_name,
    ib.code AS machine_code, ib.machine_name, ib.serial_number,
    st.assigned_to, st.reported_date
FROM service_tickets st
JOIN customers c ON st.customer_id = c.id
LEFT JOIN installed_base ib ON st.installed_base_id = ib.id
WHERE st.status NOT IN ('closed','cancelled');

DROP VIEW IF EXISTS v_spare_parts_low_stock;
CREATE VIEW IF NOT EXISTS v_spare_parts_low_stock AS
SELECT
    sp.id AS part_id,
    sp.part_number, sp.name, sp.category,
    sps.quantity, sps.min_stock,
    (sps.quantity - sps.min_stock) AS margin,
    sp.supplier, sp.unit_cost, sp.is_critical
FROM spare_parts sp
JOIN spare_part_stock sps ON sp.id = sps.part_id
WHERE sps.quantity <= sps.min_stock * 1.2
ORDER BY margin ASC;

DROP VIEW IF EXISTS v_active_alerts;
CREATE VIEW IF NOT EXISTS v_active_alerts AS
SELECT
    ma.id, ma.machine_id, ma.alert_type, ma.severity, ma.value, ma.threshold,
    ma.message, ma.timestamp,
    ib.code AS machine_code, ib.machine_name,
    c.name AS customer_name
FROM machine_alerts ma
JOIN installed_base ib ON ma.machine_id = ib.id
JOIN customers c ON ib.customer_id = c.id
WHERE ma.resolved = 0
ORDER BY
    CASE ma.severity
        WHEN 'emergency' THEN 0
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        ELSE 3
    END,
    ma.timestamp DESC;

DROP VIEW IF EXISTS v_installed_base_overview;
CREATE VIEW IF NOT EXISTS v_installed_base_overview AS
SELECT
    ib.id,
    ib.customer_id,
    ib.project_id,
    ib.code, ib.machine_name, ib.serial_number, ib.status,
    ib.installation_date, ib.warranty_expiry,
    ib.location_plant, ib.location_line,
    c.name AS customer_name, c.country,
    pm.model_code, pf.name AS family_name
FROM installed_base ib
JOIN customers c ON ib.customer_id = c.id
LEFT JOIN product_models pm ON ib.model_id = pm.id
LEFT JOIN product_families pf ON pm.family_id = pf.id
ORDER BY ib.installation_date DESC;
