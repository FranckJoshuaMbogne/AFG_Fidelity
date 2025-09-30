# streamlit_app_live.py
from __future__ import annotations
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------
# Configuration
# ---------------------------
TIERS: Dict[str, int] = {"Palier Bronze": 100, "Palier Argent": 200, "Palier Or": 500}
DEFAULT_POINTS_PER_CURRENCY = 1.0  # (non utilisé pour la distribution fixe ci-dessous)

# Distribution des types de transactions et points attribués (fixe par transaction)
TX_TYPE_POINTS = {
    "B2W / W2B": 1,                          # B2W/W2B
    "Opération d'épargne": 2,               # Opération d’épargne
    "Achat en ligne": 3,                    # Achat en ligne
    "Retrait par carte au GAB": 2,          # Retrait par carte au GAB
    "Paiement par carte au TPE": 2,         # Paiement par carte au TPE
    "Virement en local": 1,                 # Virement en local
    "Mini relevé bancaire": 1,              # Mini relevé bancaire
    "Transfert en interne": 1,              # Transfert en interne
    "Paiement de facture": 1,               # Paiement de facture
    "Parrainage d'un utilisateur": 5        # Parrainage d’un utilisateur
}

TX_TYPES = list(TX_TYPE_POINTS.keys())

# ---------------------------
# Fonctions utilitaires
# ---------------------------

def init_sample_clients(n: int = 12, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    names = [
        "Aisha Kamoise", "Bruno Bidjan", "Chantal BIYA", "David Eto'o", "Estelle MOUKOKO", "Franck MBOGNE",
        "Maxime FOGUE", "Arlette FOSSO", "Isabelle Vianey", "Jean Luc", "Koffi Annan", "Ingrid ABOMO"
    ]
    contacts = [
        "aisha@example.com", "bruno@example.com", "chantal@example.com", "david@example.com",
        "estelle@example.com", "franck@example.com", "maxime@example.com", "arlette@example.com",
        "isabelle@example.com", "jean@example.com", "koffi@example.com", "ingrid@example.com"
    ]
    clients = []
    now = datetime.now()
    for i in range(n):
        starting_points = int(abs(np.random.normal(loc=40, scale=60)))
        clients.append({
            "client_id": f"C{i+1:03d}",
            "name": names[i % len(names)],
            "email": contacts[i % len(contacts)],
            "points": int(starting_points),
            "created_at": (now - timedelta(days=random.randint(30, 365))).isoformat()
        })
    return pd.DataFrame(clients)


def generate_transaction(client_id: str) -> dict:
    """Simule une transaction : montant, type, points attribués, timestamp."""
    amount = max(1, int(np.random.exponential(scale=40)))
    tx_type = random.choice(TX_TYPES)
    points_awarded = TX_TYPE_POINTS[tx_type]
    return {
        "client_id": client_id,
        "amount": int(amount),
        "tx_type": tx_type,
        "points_awarded": int(points_awarded),
        "timestamp": datetime.now().isoformat()
    }


def apply_transaction(df_clients: pd.DataFrame, tx: dict) -> pd.DataFrame:
    """Applique une transaction en ajoutant les points_awarded (fixe selon le type)."""
    idx = df_clients.index[df_clients["client_id"] == tx["client_id"]]
    if len(idx) == 0:
        return df_clients
    i = idx[0]
    added_points = int(tx.get("points_awarded", 0))
    df_clients.at[i, "points"] = int(df_clients.at[i, "points"]) + added_points
    return df_clients


def current_tier(points: int) -> Tuple[str, Tuple[str, int]]:
    tiers_sorted = sorted(TIERS.items(), key=lambda x: x[1])
    cur = None
    next_t = None
    for name, needed in tiers_sorted:
        if points >= needed:
            cur = name
    for name, needed in tiers_sorted:
        if points < needed:
            next_t = (name, needed - points)
            break
    if cur is None:
        cur = "Aucun"
    if next_t is None:
        next_t = ("--", 0)
    return cur, next_t

# ---------------------------
# Streamlit : UI & logique live
# ---------------------------

st.set_page_config(page_title="Simulateur AFG Fidelity", layout="wide")
st.title("Simulateur AFG Fidelity")

# Sidebar : paramètres
with st.sidebar.expander("Paramètres de simulation", expanded=True):
    seed = st.number_input("Seed (répétable)", min_value=0, max_value=99999, value=42, step=1)
    n_clients = st.slider("Nombre de clients", 4, 100, value=12, step=1)
    total_ticks = st.number_input("Nombre d'étapes de la simulation", min_value=1, max_value=5000, value=200, step=1)
    mode = st.radio("Mode d'exécution", ("Auto", "Pas-à-pas"))
    speed = st.slider("Vitesse (s délai entre ticks)", 0.0, 2.0, 0.3, step=0.05)

# Initialisation dans session_state (persistant entre re-runs)
if "initialized" not in st.session_state or st.session_state.get("seed") != seed or st.session_state.get("n_clients") != n_clients:
    st.session_state["clients_df"] = init_sample_clients(n=n_clients, seed=int(seed))
    st.session_state["tick"] = 0
    st.session_state["transactions"] = []     # liste dicts: client_id, name, tx_type, amount, points_awarded, timestamp
    st.session_state["history_top"] = []      # conserve topN au fil du temps
    st.session_state["running"] = False
    st.session_state["initialized"] = True
    st.session_state["seed"] = seed
    st.session_state["n_clients"] = n_clients

clients_df: pd.DataFrame = st.session_state["clients_df"]

# Contrôles d'exécution (Start / Stop / Reset / Step)
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1,1,1])
with col_ctrl1:
    if st.button("Démarrer (Auto)"):
        st.session_state["running"] = True
        st.session_state["mode"] = "Auto"
with col_ctrl2:
    if st.button("Stop"):
        st.session_state["running"] = False
with col_ctrl3:
    if st.button("Réinitialiser"):
        st.session_state["clients_df"] = init_sample_clients(n=n_clients, seed=int(seed))
        st.session_state["tick"] = 0
        st.session_state["transactions"] = []
        st.session_state["history_top"] = []
        st.session_state["running"] = False
        st.rerun()

# Bouton pas-à-pas (unique tick)
if mode == "Pas-à-pas":
    if st.button("Avancer d'un tick"):
        st.session_state["mode"] = "Pas-à-pas"
        st.session_state["running"] = False
        do_step = True
    else:
        do_step = False
else:
    do_step = False

# Placeholders (mise à jour à chaque tick)
placeholder_table = st.empty()
placeholder_chart = st.empty()
placeholder_info = st.empty()
placeholder_progress = st.progress(0)
placeholder_transactions = st.empty()  # pour le tableau "Transactions du mois"

# Fonction pour exécuter un tick et mettre à jour l'état
def run_single_tick():
    clients_df_local: pd.DataFrame = st.session_state["clients_df"]
    # on génère entre 1 et 3 transactions par tick (configurable)
    n_tx = random.randint(1, 3)
    txs = []
    changed = {}
    for _ in range(n_tx):
        client = random.choice(clients_df_local["client_id"].tolist())
        tx = generate_transaction(client)
        # récupérer le nom du client pour l'enregistrement des transactions
        client_name = clients_df_local.loc[clients_df_local["client_id"] == client, "name"].values[0]
        # ajouter points
        before = int(clients_df_local.loc[clients_df_local["client_id"] == client, "points"].values[0])
        clients_df_local = apply_transaction(clients_df_local, tx)
        after = int(clients_df_local.loc[clients_df_local["client_id"] == client, "points"].values[0])
        changed[client] = after - before

        # Enregistrer la transaction complète dans session_state["transactions"]
        tx_record = {
            "timestamp": tx["timestamp"],
            "client_id": client,
            "name": client_name,
            "tx_type": tx["tx_type"],
            "amount": tx["amount"],
            "points_awarded": tx["points_awarded"]
        }
        txs.append(tx_record)

    st.session_state["transactions"].extend(txs)
    st.session_state["clients_df"] = clients_df_local
    st.session_state["tick"] += 1

    # update history top (top 5)
    top5 = clients_df_local.sort_values("points", ascending=False).head(5)[["client_id","name","points"]]
    st.session_state["history_top"].append({
        "tick": st.session_state["tick"],
        "top5": top5.to_dict(orient="records"),
    })
    st.session_state["last_n_tx"] = n_tx
    return changed

# Exécution principale (mode Auto: boucle; mode Pas-à-pas: 1 tick)
max_ticks = int(total_ticks)
if mode == "Auto" and st.session_state.get("running", False):
    # boucle jusqu'à max_ticks ou arrêt utilisateur
    while st.session_state["tick"] < max_ticks and st.session_state["running"]:
        changed_map = run_single_tick()

        # Affichage
        df_display = st.session_state["clients_df"].copy()
        df_display["rank"] = df_display["points"].rank(method="min", ascending=False).astype(int)
        df_display = df_display.sort_values(["points","name"], ascending=[False, True]).reset_index(drop=True)

        # On met en évidence le changement: ajoute une colonne delta si existant
        df_display["delta"] = df_display["client_id"].map(changed_map).fillna(0).astype(int)

        # Table interactif
        with placeholder_table.container():
            st.subheader(f"Tick {st.session_state['tick']} / {max_ticks}")
            display_df = df_display[["rank","client_id","name","points","delta"]].copy()
            display_df = display_df.rename(columns={"rank":"Classement","client_id":"ID","name":"Nom","points":"Points","delta":"Δ points"})
            st.table(display_df)

        # Bar chart du classement
        with placeholder_chart.container():
            st.subheader("Classement")
            chart_df = df_display.set_index("name")["points"]
            st.bar_chart(chart_df)

        # Progression
        pct = min(1.0, st.session_state["tick"] / max_ticks)
        placeholder_progress.progress(pct)

        # Info latérale
        with placeholder_info.container():
            last_n_tx = st.session_state.get("last_n_tx", 0)
            st.caption(f"Transactions ce tick: {last_n_tx} (générées aléatoirement). Total transactions: {len(st.session_state['transactions'])}")
            # Afficher top 3
            top3 = df_display.head(3)[["name","points"]].to_dict(orient="records")
            for i, t in enumerate(top3, start=1):
                st.metric(label=f"Top {i}", value=f"{t['name']}", delta=f"{t['points']} pts")

        # Affichage du tableau Transactions du mois et bouton download
        with placeholder_transactions.container():
            st.subheader("Transactions du mois")
            if len(st.session_state["transactions"]) == 0:
                st.info("Aucune transaction pour le moment.")
            else:
                tx_df = pd.DataFrame(st.session_state["transactions"])
                # Ordre: timestamp, name, tx_type, amount, points_awarded
                tx_df = tx_df[["timestamp","name","tx_type","amount","points_awarded"]].rename(columns={
                    "timestamp":"Horodatage",
                    "name":"Nom client",
                    "tx_type":"Type transaction",
                    "amount":"Montant",
                    "points_awarded":"Points attribués"
                })
                # Afficher le tableau (restreint à 200 lignes affichées pour perf)
                st.dataframe(tx_df.head(200))
                # Préparer CSV pour téléchargement
                csv_bytes = tx_df.to_csv(index=False).encode("utf-8")
                # st.download_button("Télécharger les transactions (CSV)", data=csv_bytes, file_name="transactions_du_mois.csv", mime="text/csv")

        # Attente (contrôlable)
        time.sleep(float(speed))

        # petit check : si utilisateur a cliqué Stop via bouton, session_state["running"] devient False et boucle s'arrête
        if not st.session_state["running"]:
            break

    # fin loop
    if st.session_state["tick"] >= max_ticks:
        st.session_state["running"] = False
        st.success("Simulation terminée (auto).")

else:
    # Mode 'Pas-à-pas' ou 'Auto' mais non démarré: afficher état actuel et transactions
    df_display = st.session_state["clients_df"].copy()
    df_display["rank"] = df_display["points"].rank(method="min", ascending=False).astype(int)
    df_display = df_display.sort_values(["points","name"], ascending=[False, True]).reset_index(drop=True)
    df_display["delta"] = 0
    display_df = df_display[["rank","client_id","name","points","delta"]].copy()
    display_df = display_df.rename(columns={"rank":"Classement","client_id":"ID","name":"Nom","points":"Points","delta":"Δ points"})
    placeholder_table.table(display_df)
    placeholder_chart.bar_chart(df_display.set_index("name")["points"])
    placeholder_progress.progress(min(1.0, st.session_state["tick"] / max_ticks if max_ticks>0 else 0.0))
    placeholder_info.caption(f"Tick: {st.session_state['tick']} - Appuie sur 'Démarrer (Auto)' ou 'Avancer d'un tick' pour voir l'évolution.")

    # Transactions du mois affichées en mode inactif également
    with placeholder_transactions.container():
        st.subheader("Transactions du mois")
        if len(st.session_state["transactions"]) == 0:
            st.info("Aucune transaction pour le moment.")
        else:
            tx_df = pd.DataFrame(st.session_state["transactions"])
            tx_df = tx_df[["timestamp","name","tx_type","amount","points_awarded"]].rename(columns={
                "timestamp":"Horodatage",
                "name":"Nom client",
                "tx_type":"Type transaction",
                "amount":"Montant",
                "points_awarded":"Points attribués"
            })
            st.dataframe(tx_df.head(200))
            csv_bytes = tx_df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les transactions (CSV)", data=csv_bytes, file_name="transactions_du_mois.csv", mime="text/csv")

# Si le mode est Pas-à-pas et qu'on a demandé un pas, exécuter 1 tick puis re-run pour rafraîchir l'UI
if mode == "Pas-à-pas" and do_step:
    changed_map = run_single_tick()
    st.rerun()
