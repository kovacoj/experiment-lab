from __future__ import annotations

REPUTATION_LABELS = [
    "slow_service",
    "queue_or_waiting",
    "rude_staff",
    "positive_service",
    "price_complaint",
    "competitor_discount",
    "new_menu",
    "oat_milk_trend",
    "morning_peak",
]

SUPPLY_CHAIN_LABELS = [
    "lead_time_increase",
    "inventory_shortage",
    "price_increase",
    "shipping_delay",
    "supplier_disruption",
    "alternative_supplier_available",
    "geopolitical_risk",
    "commodity_price_pressure",
]

ENTITY_ALIASES = {
    "reputation_monitor": {
        "Miners Vinohrady": ("Miners Vinohrady", "location"),
        "Miners Wenceslas": ("Miners Wenceslas", "location"),
        "Miners Karlin": ("Miners Karlin", "location"),
        "Miners Letna": ("Miners Letna", "location"),
        "Bean House": ("Bean House", "competitor"),
        "Brew Yard": ("Brew Yard", "competitor"),
        "North Cup": ("North Cup", "competitor"),
        "oat milk": ("oat milk", "menu_item"),
        "matcha": ("matcha", "menu_item"),
    },
    "supply_chain_risk": {
        "MCU chip": ("MCU chip", "component"),
        "lithium battery pack": ("lithium battery pack", "component"),
        "Hamburg": ("Hamburg", "port"),
        "Taiwan/Asia": ("Taiwan/Asia", "region"),
        "Euro Silicon Backup": ("Euro Silicon Backup", "supplier"),
        "Nordic Logic Backup": ("Nordic Logic Backup", "supplier"),
    },
}
