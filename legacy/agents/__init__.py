"""
Domain agent exports for the Coiltech Industrial Automation Platform.
"""

from agents.application_engineering_agent import ApplicationEngineeringAgent
from agents.base_agent import AgentResponse, BaseAgent, GeneralAgent
from agents.configuration_agent import ConfigurationAgent
from agents.digital_twin_agent import DigitalTwinAgent
from agents.installation_commissioning_agent import InstallationCommissioningAgent
from agents.management_reporting_agent import ManagementReportingAgent
from agents.orchestrator import Orchestrator
from agents.product_catalog_agent import ProductCatalogAgent
from agents.project_execution_agent import ProjectExecutionAgent
from agents.quality_compliance_agent import QualityComplianceAgent
from agents.quotation_agent import QuotationAgent
from agents.router_agent import IntentResult, RouterAgent
from agents.service_maintenance_agent import ServiceMaintenanceAgent
from agents.spare_parts_agent import SparePartsAgent

__all__ = [
    "AgentResponse",
    "ApplicationEngineeringAgent",
    "BaseAgent",
    "ConfigurationAgent",
    "DigitalTwinAgent",
    "GeneralAgent",
    "InstallationCommissioningAgent",
    "IntentResult",
    "ManagementReportingAgent",
    "Orchestrator",
    "ProductCatalogAgent",
    "ProjectExecutionAgent",
    "QualityComplianceAgent",
    "QuotationAgent",
    "RouterAgent",
    "ServiceMaintenanceAgent",
    "SparePartsAgent",
]
