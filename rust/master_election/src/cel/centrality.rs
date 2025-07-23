use crate::cel::data::Map;
use crate::cel::{View, ID};

/// The number of neighbours of a process.
pub fn degree_centrality(known: &Map<ID, View>, id: ID) -> u32 {
    todo!()
}

/// Measures average length of the shortest path between the vertex and all other vertices in the graph.
/// The more central is a vertex, the closer it is to all other vertices. The closeness centrality
/// characterizes the ability of a node to spread information over the graph.
///
/// Alex Balevas defined in 1950 the closeness centrality of a vertex as follows:
/// `C_C(x) = \frac{1}{ \sum_y d(x,y) }` where `d(x,y)` is the shortest path between `x` and `y`.
///
/// CEL paper uses this.
pub fn closeness_centrality(known: &Map<ID, View>, id: ID) -> u32 {
    todo!()
}

/// Measures the number of times a vertex acts as a relay (router) along
/// shortest paths between other vertices. Even if previous authors
/// have intuitively described centrality as being based on betweenness,
/// betweenness centrality was formally defined by Freeman in 1977.
///
/// The betweenness of a vertex `x` is defined as the sum, for each pair
/// of vertices `(s, t)`, of the number of shortest paths from `s` to `t` that
/// pass through `x`, over the total number of shortest paths between
/// vertices `s` and `t`; it can be represented by the following formula:
/// `C_B(x) = \sum_{ s \neq x \neq t } \frac{ \sigma_{st}(x) }{ \sigma_{st} }`
/// where `\sigma_{st}` denotes the total number of shortest paths from vertex `s`
/// to vertex `t` (with `\sigma_{ss} = 1` by convention), and `\sigma_{st}(x)`
/// is the number of those shorter paths that pass through `x`.
pub fn betweenness_centrality(known: &Map<ID, View>, id: ID) -> u32 {
    todo!()
}
