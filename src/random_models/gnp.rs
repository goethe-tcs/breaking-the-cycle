use crate::graph::*;
use rand::Rng;
use rand_distr::Geometric;

/// Provides an iterator similarly to Range, but
/// includes each element i.i.d. with probability of p
pub struct BernoulliSamplingRange<'a, R: Rng> {
    current: i64,
    end: i64,
    distr: Geometric,
    rng: &'a mut R,
}

impl<'a, R: Rng> BernoulliSamplingRange<'a, R> {
    pub fn new(rng: &'a mut R, begin: i64, end: i64, prob: f64) -> Self {
        assert!(begin <= end);
        assert!((0.0..=1.0).contains(&prob));
        Self {
            rng,
            current: begin - 1,
            end,
            distr: Geometric::new(prob).unwrap(),
        }
    }

    fn try_advance(&mut self) {
        if self.current >= self.end {
            return;
        }

        let skip = self.rng.sample(self.distr);
        if skip > i64::MAX as u64 {
            self.current = self.end;
        } else {
            self.current += 1;
            self.current = match self.current.checked_add(skip as i64) {
                Some(x) => x,
                None => self.end,
            }
        }
    }
}

impl<'a, R: Rng> Iterator for BernoulliSamplingRange<'a, R> {
    type Item = i64;
    fn next(&mut self) -> Option<Self::Item> {
        self.try_advance();

        if self.current >= self.end {
            None
        } else {
            Some(self.current)
        }
    }
}

/// Generates a Gilbert (also, wrongly, known as Erdos-Reyni) graph
/// The G(n,p) contains n nodes and each of the n^2 edges exists
/// independently with probability p
pub fn generate_gnp<G, R>(rng: &mut R, n: Node, p: f64) -> G
where
    R: Rng,
    G: GraphNew + GraphEdgeEditing,
{
    let mut result = G::new(n as usize);

    let iter = BernoulliSamplingRange::new(rng, 0, (n as i64) * (n as i64), p);
    for x in iter {
        let u = x / (n as i64);
        let v = x % (n as i64);
        result.add_edge(u as Node, v as Node);
    }

    result
}

#[cfg(test)]
mod test {
    use crate::graph::{AdjListMatrix, GraphOrder};
    use crate::random_models::gnp::{generate_gnp, BernoulliSamplingRange};

    #[test]
    fn test_bernoulli_range() {
        let rng = &mut rand::thread_rng();

        // empty range
        assert_eq!(
            BernoulliSamplingRange::new(rng, 0, 0, 1.0)
                .into_iter()
                .count(),
            0
        );

        // p=1
        assert_eq!(
            BernoulliSamplingRange::new(rng, 0, 10, 1.0)
                .into_iter()
                .count(),
            10
        );

        // p=0
        assert_eq!(
            BernoulliSamplingRange::new(rng, 0, 100, 0.0)
                .into_iter()
                .count(),
            0
        );

        // test that we see each element ~p*n times
        let min = 3;
        let max = 100;
        let mut counts = vec![0; max as usize];
        for _ in 0..1000 {
            let b = BernoulliSamplingRange::new(rng, min, max, 0.25);
            for x in b {
                assert!(min <= x);
                assert!(x < max);
                counts[x as usize] += 1;
            }
        }

        assert!(counts.iter().enumerate().all(|(i, &c)| {
            if i < min as usize {
                c == 0
            } else {
                (150..350).contains(&c)
            }
        }));
    }

    #[test]
    fn test_gnp() {
        let rng = &mut rand::thread_rng();

        // generate multiple graphs of various densities and verify that the
        // expected number of edges is close to the expected value
        for p in [0.001, 0.01, 0.1] {
            let repeats = 100;
            let n = 100;

            let mean_edges = (0..repeats)
                .into_iter()
                .map(|_| {
                    let g: AdjListMatrix = generate_gnp(rng, n, p);
                    g.number_of_edges() as f64
                })
                .sum::<f64>()
                / repeats as f64;

            let expected = p * (n as f64).powi(2);

            assert!((0.75 * expected..1.25 * expected).contains(&mean_edges));
        }
    }
}
