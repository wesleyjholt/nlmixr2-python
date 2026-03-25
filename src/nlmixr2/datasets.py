"""Built-in pharmacometric example datasets.

Provides synthetic datasets modeled after standard pharmacometric reference data,
equivalent to R's nlmixr2data package. All data is generated deterministically
using known PK parameters with fixed seeds.
"""

import math
import random


def _one_comp_oral(dose, ka, ke, v, times):
    """Simulate one-compartment oral PK model concentrations."""
    conc = []
    for t in times:
        if t <= 0:
            conc.append(0.0)
        else:
            c = (dose * ka / (v * (ka - ke))) * (math.exp(-ke * t) - math.exp(-ka * t))
            conc.append(max(c, 0.0))
    return conc


def _one_comp_iv(dose, ke, v, times_since_dose):
    """Simulate one-compartment IV bolus PK model concentration at times after dose."""
    conc = []
    for t in times_since_dose:
        if t <= 0:
            conc.append(0.0)
        else:
            c = (dose / v) * math.exp(-ke * t)
            conc.append(max(c, 0.0))
    return conc


def theo_sd():
    """Theophylline single-dose PK data (synthetic).

    Modeled after the classic Boeckmann, Sheiner & Beal Theophylline dataset.
    12 subjects receive a single oral dose (~4-5 mg/kg) with ~10 observations each.

    Returns:
        dict of lists with keys: id, time, dv, amt, evid, wt
    """
    rng = random.Random(42)

    # Typical PK parameters for Theophylline oral
    # ka ~ 1.5 1/hr, ke ~ 0.08 1/hr, V/F ~ 0.5 L/kg
    obs_times = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0]

    ids = []
    times = []
    dvs = []
    amts = []
    evids = []
    wts = []

    for subj in range(1, 13):
        wt = round(rng.gauss(70, 10), 1)
        wt = max(50.0, min(wt, 95.0))
        dose = round(wt * rng.uniform(4.0, 5.5), 1)  # mg, ~4-5.5 mg/kg

        # Individual PK parameters with inter-individual variability
        ka = 1.5 * math.exp(rng.gauss(0, 0.3))
        ke = 0.08 * math.exp(rng.gauss(0, 0.25))
        v = 0.5 * wt * math.exp(rng.gauss(0, 0.2))

        # Dosing record at time 0
        ids.append(subj)
        times.append(0.0)
        dvs.append(0.0)
        amts.append(dose)
        evids.append(1)
        wts.append(wt)

        # Simulate observations
        concs = _one_comp_oral(dose, ka, ke, v, obs_times)
        for j, t in enumerate(obs_times):
            c = concs[j]
            # Add proportional + additive residual error
            c_obs = c * math.exp(rng.gauss(0, 0.1)) + rng.gauss(0, 0.1)
            c_obs = round(max(c_obs, 0.01), 2)
            ids.append(subj)
            times.append(t)
            dvs.append(c_obs)
            amts.append(0)
            evids.append(0)
            wts.append(wt)

    return {
        "id": ids,
        "time": times,
        "dv": dvs,
        "amt": amts,
        "evid": evids,
        "wt": wts,
    }


def warfarin():
    """Warfarin PK data (synthetic).

    Modeled after the classic Warfarin PK dataset used in NONMEM examples.
    ~32 subjects receive a single oral dose with multiple observations.

    Returns:
        dict of lists with keys: id, time, dv, amt, evid, wt, age, sex
    """
    rng = random.Random(123)

    # Warfarin: ka ~ 1.0 1/hr, ke ~ 0.03 1/hr, V/F ~ 8 L
    obs_times_base = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0, 96.0, 120.0]

    ids = []
    times = []
    dvs = []
    amts = []
    evids = []
    wts = []
    ages = []
    sexes = []

    for subj in range(1, 33):
        sex = rng.choice([0, 1])
        age = rng.randint(22, 75)
        wt = round(rng.gauss(75 if sex == 0 else 65, 12), 1)
        wt = max(45.0, min(wt, 120.0))
        dose = rng.choice([5.0, 7.5, 10.0])  # mg warfarin

        # Individual PK parameters
        ka = 1.0 * math.exp(rng.gauss(0, 0.35))
        ke = 0.03 * math.exp(rng.gauss(0, 0.3))
        v = 8.0 * math.exp(rng.gauss(0, 0.25))

        # Some subjects may miss a timepoint
        subj_times = [t for t in obs_times_base if rng.random() > 0.1]

        # Dosing record
        ids.append(subj)
        times.append(0.0)
        dvs.append(0.0)
        amts.append(dose)
        evids.append(1)
        wts.append(wt)
        ages.append(age)
        sexes.append(sex)

        # Observations
        concs = _one_comp_oral(dose, ka, ke, v, subj_times)
        for j, t in enumerate(subj_times):
            c = concs[j]
            c_obs = c * math.exp(rng.gauss(0, 0.12)) + rng.gauss(0, 0.02)
            c_obs = round(max(c_obs, 0.01), 3)
            ids.append(subj)
            times.append(t)
            dvs.append(c_obs)
            amts.append(0)
            evids.append(0)
            wts.append(wt)
            ages.append(age)
            sexes.append(sex)

    return {
        "id": ids,
        "time": times,
        "dv": dvs,
        "amt": amts,
        "evid": evids,
        "wt": wts,
        "age": ages,
        "sex": sexes,
    }


def pheno_sd():
    """Phenobarbital neonatal PK data (synthetic).

    Modeled after the classic Phenobarbital dataset (Grasela & Donn, 1985).
    ~59 neonatal subjects receiving IV phenobarbital with sparse sampling.

    Returns:
        dict of lists with keys: id, time, dv, amt, evid, wt, apgr
    """
    rng = random.Random(789)

    ids = []
    times = []
    dvs = []
    amts = []
    evids = []
    wts = []
    apgrs = []

    for subj in range(1, 60):
        wt = round(rng.gauss(1.5, 0.5), 2)
        wt = max(0.6, min(wt, 3.5))
        apgr = rng.randint(1, 10)

        # Phenobarbital: typical Cl ~ 0.005 L/hr/kg, V ~ 1.0 L/kg
        v = 1.0 * wt * math.exp(rng.gauss(0, 0.25))
        cl = 0.005 * wt * math.exp(rng.gauss(0, 0.3))
        ke = cl / v

        # Neonates get multiple IV doses; simulate 1-3 doses with observations
        n_doses = rng.randint(1, 3)
        dose_times_list = sorted(rng.sample([0.0, 12.0, 24.0, 48.0][:n_doses + 1], n_doses))
        dose_amt = round(wt * rng.uniform(15, 25), 1)  # mg, ~15-25 mg/kg loading

        # Build subject records: doses and observations
        subject_records = []

        # Add dosing records
        for dt in dose_times_list:
            subject_records.append({
                "time": dt,
                "dv": 0.0,
                "amt": dose_amt,
                "evid": 1,
            })

        # Add observation records (sparse: 1-4 obs per subject)
        n_obs = rng.randint(1, 4)
        last_dose_time = dose_times_list[-1]
        possible_obs = []
        for dt in dose_times_list:
            for offset in [2.0, 6.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]:
                obs_t = dt + offset
                if obs_t > dt and obs_t not in dose_times_list:
                    possible_obs.append(obs_t)
        possible_obs = sorted(set(possible_obs))
        if len(possible_obs) > n_obs:
            obs_sample = sorted(rng.sample(possible_obs, n_obs))
        else:
            obs_sample = possible_obs

        # Superposition: compute concentration from all previous doses
        for obs_t in obs_sample:
            total_conc = 0.0
            for dt in dose_times_list:
                if obs_t > dt:
                    elapsed = obs_t - dt
                    total_conc += (dose_amt / v) * math.exp(-ke * elapsed)
            c_obs = total_conc * math.exp(rng.gauss(0, 0.15)) + rng.gauss(0, 0.5)
            c_obs = round(max(c_obs, 0.1), 2)
            subject_records.append({
                "time": obs_t,
                "dv": c_obs,
                "amt": 0,
                "evid": 0,
            })

        # Sort by time
        subject_records.sort(key=lambda r: (r["time"], -r["evid"]))

        for rec in subject_records:
            ids.append(subj)
            times.append(rec["time"])
            dvs.append(rec["dv"])
            amts.append(rec["amt"])
            evids.append(rec["evid"])
            wts.append(wt)
            apgrs.append(apgr)

    return {
        "id": ids,
        "time": times,
        "dv": dvs,
        "amt": amts,
        "evid": evids,
        "wt": wts,
        "apgr": apgrs,
    }


_DATASET_REGISTRY = {
    "theo_sd": theo_sd,
    "warfarin": warfarin,
    "pheno_sd": pheno_sd,
}


def list_datasets():
    """Return a list of available dataset names.

    Returns:
        list of str
    """
    return sorted(_DATASET_REGISTRY.keys())


def load_dataset(name):
    """Load a dataset by name.

    Args:
        name: Dataset name (use list_datasets() to see available names).

    Returns:
        dict of lists

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return _DATASET_REGISTRY[name]()
