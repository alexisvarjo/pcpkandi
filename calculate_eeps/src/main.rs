use csv::{ReaderBuilder, WriterBuilder};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::error::Error;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    #[serde(rename = "Date")]
    date: String,
    y: f64,
    x: f64,
    call_v: f64,
    put_v: f64,
    ulying_div: f64,
    ulying_volume: f64,
    strike: f64,
    maturity: f64,
    call_price: f64,
    put_price: f64,
    ulying_price: f64,
    risk_free_rate: f64,
    put_moneyness: f64,
    call_moneyness: f64,
    #[serde(rename = "IV_put")]
    iv_put: f64,
    #[serde(rename = "IV_call")]
    iv_call: f64,
    country: String,
    #[serde(rename = "PV_alldivs")]
    pv_alldivs: f64,
    underlying_return: Option<f64>,
    underlying_log_return: Option<f64>,
    underlying_volatility: Option<f64>,
    ulying_illiquidity: Option<f64>,
    put_illiquidity: Option<f64>,
    call_illiquidity: Option<f64>,
    // Additional computed columns; default ensures that missing CSV values are allowed.
    #[serde(default)]
    eep_call: Option<f64>,
    #[serde(default)]
    eep_put: Option<f64>,
}

/// Prices an American option using a Crank–Nicolson finite difference method with PSOR.
fn american_option_price_fd(
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    smax: f64,
    m: usize,
    n: usize,
    option_type: &str,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let d_s = smax / m as f64;
    let dt = t / n as f64;
    // Create the asset price grid: 0, d_s, 2*d_s, ..., smax
    let s_values: Vec<f64> = (0..=m).map(|i| i as f64 * d_s).collect();

    let (payoff, boundary_low, boundary_high, min_value) = if option_type == "put" {
        (
            s_values
                .iter()
                .map(|&s| (k - s).max(0.0))
                .collect::<Vec<f64>>(),
            k,
            0.0,
            (k - s0).max(0.0),
        )
    } else {
        (
            s_values
                .iter()
                .map(|&s| (s - k).max(0.0))
                .collect::<Vec<f64>>(),
            0.0,
            smax - k,
            (s0 - k).max(0.0),
        )
    };

    let mut v = vec![vec![0.0; m + 1]; n + 1];
    v[n] = payoff.clone();
    for j in 0..=n {
        v[j][0] = boundary_low;
        v[j][m] = boundary_high;
    }

    for j in (0..n).rev() {
        v[j] = v[j + 1].clone();
        let mut error = 1.0;
        let mut iter_count = 0;
        while error > tol && iter_count < max_iter {
            error = 0.0;
            for i in 1..m {
                let s_i = i as f64 * d_s;
                let alpha = 0.25 * dt * (sigma * sigma * (i as f64 * i as f64) - r * i as f64);
                let beta = -0.5 * dt * (sigma * sigma * (i as f64 * i as f64) + r);
                let gamma = 0.25 * dt * (sigma * sigma * (i as f64 * i as f64) + r * i as f64);
                let a = -alpha;
                let b = 1.0 - beta;
                let c = -gamma;
                let d =
                    alpha * v[j + 1][i - 1] + (1.0 + beta) * v[j + 1][i] + gamma * v[j + 1][i + 1];
                let v_old = v[j][i];
                let new_val =
                    (1.0 - omega) * v_old + (omega / b) * (d - a * v[j][i - 1] - c * v[j][i + 1]);
                let intrinsic = if option_type == "put" {
                    k - s_i
                } else {
                    s_i - k
                };
                let v_new = new_val.max(intrinsic);
                error = error.max((v_new - v_old).abs());
                v[j][i] = v_new;
            }
            iter_count += 1;
        }
    }

    let idx = (s0 / d_s).floor() as usize;
    let idx = if idx >= m { m - 1 } else { idx };
    let s_lower = s_values[idx];
    let s_upper = s_values[idx + 1];
    let v_lower = v[0][idx];
    let v_upper = v[0][idx + 1];
    let option_price = v_lower + (v_upper - v_lower) * ((s0 - s_lower) / (s_upper - s_lower));
    option_price.max(min_value)
}

/// Computes the European option price via the Black–Scholes formula.
fn compute_euro_option_bs(
    s_adj: f64,
    _pv_divs: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: &str,
) -> f64 {
    let d1 = ((s_adj / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    let norm = Normal::new(0.0, 1.0).unwrap();
    if option_type == "call" {
        s_adj * norm.cdf(d1) - k * (-r * t).exp() * norm.cdf(d2)
    } else if option_type == "put" {
        k * (-r * t).exp() * norm.cdf(-d2) - s_adj * norm.cdf(-d1)
    } else {
        panic!("Invalid option type");
    }
}

/// Computes the early exercise premium (EEP) for a given record using finite differences.
fn compute_early_exercise_premium_fd(record: &Record, m: usize, n: usize) -> (f64, f64) {
    let s = record.ulying_price;
    let k = record.strike;
    let t = record.maturity;
    let r = record.risk_free_rate;
    let sigma = (record.iv_put + record.iv_call) / 2.0;
    let pv_div = record.pv_alldivs;

    let s_adj = s - pv_div;
    if s_adj <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let smax = (2.0 * s_adj).max(2.0 * k);
    let a_call =
        american_option_price_fd(s_adj, k, t, r, sigma, smax, m, n, "call", 1.2, 1e-3, 10000);
    let a_put =
        american_option_price_fd(s_adj, k, t, r, sigma, smax, m, n, "put", 1.2, 1e-3, 10000);
    let e_call = compute_euro_option_bs(s_adj, pv_div, k, t, r, sigma, "call");
    let e_put = compute_euro_option_bs(s_adj, pv_div, k, t, r, sigma, "put");

    let eep_call = (a_call - e_call).max(0.0);
    let eep_put = (a_put - e_put).max(0.0);
    (eep_call, eep_put)
}

/// Reads a CSV file, computes the additional columns (eep_call and eep_put), updates the x column,
/// and writes the updated records (all original columns plus the new ones) back to the same file.
fn process_file(file_path: &str) -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(file_path)?;
    let mut records: Vec<Record> = rdr.deserialize().collect::<Result<_, _>>()?;

    records.par_iter_mut().progress().for_each(|record| {
        let (eep_call, eep_put) = compute_early_exercise_premium_fd(record, 100, 100);
        record.eep_call = Some(eep_call);
        record.eep_put = Some(eep_put);
        // Update x by adding the net FD premium.
        record.x = record.x + eep_call - eep_put;
    });

    let mut wtr = WriterBuilder::new().from_path(file_path)?;
    for record in records {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Calculating EEP for Sweden");
    process_file("../processed_data/se_processed_data.csv")?;

    println!("Calculating EEP for Denmark");
    process_file("../processed_data/dk_processed_data.csv")?;

    println!("Calculating EEP for Norway");
    process_file("../processed_data/no_processed_data.csv")?;

    Ok(())
}
