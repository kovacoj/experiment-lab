from __future__ import annotations

from app.labs.reputation.competitor_price import CompetitorPriceLab
from app.labs.reputation.data_quality import DataQualityLab
from app.labs.reputation.location_sentiment import LocationSentimentLab
from app.labs.reputation.menu_trend import MenuTrendLab
from app.labs.reputation.peak_hours import PeakHoursAnalysisLab
from app.labs.reputation.staff_mention import StaffMentionLab
from app.labs.supply_chain.alternative_supplier import AlternativeSupplierLab
from app.labs.supply_chain.battery_cost import BatteryCostLab
from app.labs.supply_chain.chip_supply import ChipSupplyLab
from app.labs.supply_chain.data_quality import SupplyChainDataQualityLab
from app.labs.supply_chain.geopolitical import GeopoliticalLab
from app.labs.supply_chain.production_stop import ProductionStopRiskLab
from app.labs.supply_chain.shipping_risk import ShippingRiskLab


LAB_REGISTRY = {
    "reputation_monitor": [
        DataQualityLab,
        LocationSentimentLab,
        CompetitorPriceLab,
        PeakHoursAnalysisLab,
        MenuTrendLab,
        StaffMentionLab,
    ],
    "supply_chain_risk": [
        SupplyChainDataQualityLab,
        ChipSupplyLab,
        BatteryCostLab,
        ShippingRiskLab,
        AlternativeSupplierLab,
        GeopoliticalLab,
        ProductionStopRiskLab,
    ],
}
