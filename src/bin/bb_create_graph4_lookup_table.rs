use itertools::Itertools;

fn main() {
    let table = dfvs::exact::branch_and_bound::build_lookup_table();
    println!(
        "[{}]",
        table.iter().map(|x| format!("0x{:02x}", *x)).join(",")
    )
}
