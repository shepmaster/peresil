use super::{ParseMaster, Point, Progress, Recoverable, Status};

#[macro_export]
macro_rules! sequence {
    ($pm:expr, $pt:expr, {let $x:pat = $parser:expr; $($rest:tt)*}, $creator:expr) => {{
        let (pt, $x) = try_parse!($parser($pm, $pt));
        sequence!($pm, pt, {$($rest)*}, $creator)
    }};
    ($pm:expr, $pt:expr, {$x:pat = $parser:expr; $($rest:tt)*}, $creator:expr) => {{
        let (pt, $x) = try_parse!($parser($pm, $pt));
        sequence!($pm, pt, {$($rest)*}, $creator)
    }};
    ($pm:expr, $pt:expr, {$parser:expr; $($rest:tt)*}, $creator:expr) => {{
        let (pt, _) = try_parse!($parser($pm, $pt));
        sequence!($pm, pt, {$($rest)*}, $creator)
    }};
    ($pm:expr, $pt:expr, {}, $creator:expr) => {
        Progress::success($pt, $creator($pm, $pt))
    };
}

pub trait IntoAppend<T> {
    fn into(self) -> Vec<T>;
}

impl<T> IntoAppend<T> for Vec<T> {
    fn into(self) -> Vec<T> {
        self
    }
}

impl<T> IntoAppend<T> for Option<T> {
    fn into(self) -> Vec<T> {
        self.map(|v| vec![v]).unwrap_or_else(Vec::new)
    }
}

pub fn optional<P, E, S, F, T>(
    parser: F,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, Option<T>, E>
where
    F: FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    P: Point,
    E: Recoverable,
{
    move |pm, pt| pm.optional(pt, parser)
}

pub fn optional_append<P, E, S, A, F, T>(
    append_to: A,
    parser: F,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, Vec<T>, E>
where
    F: FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    A: IntoAppend<T>,
    P: Point,
    //E: Recoverable, // TODO: use this
{
    move |pm, pt| {
        let mut append_to = append_to.into();
        match parser(pm, pt) {
            Progress {
                point,
                status: Status::Success(v),
            } => {
                append_to.push(v);
                Progress::success(point, append_to)
            }
            Progress {
                point,
                status: Status::Failure(..),
            } => Progress::success(point, append_to),
        }
    }
}

pub fn zero_or_more<P, E, S, F, T>(
    parser: F,
) -> impl Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, Vec<T>, E>
where
    F: Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    P: Point,
    E: Recoverable,
{
    move |pm, pt| pm.zero_or_more(pt, &parser) // what why ref?
}

pub fn zero_or_more_append<P, E, S, A, F, T>(
    append_to: A,
    parser: F,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, Vec<T>, E>
where
    F: Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    A: IntoAppend<T>,
    P: Point,
    E: Recoverable,
{
    move |pm, mut pt| {
        let mut append_to = append_to.into();
        loop {
            match parser(pm, pt) {
                Progress {
                    point,
                    status: Status::Success(v),
                } => {
                    append_to.push(v);
                    pt = point;
                }
                Progress { .. } => return Progress::success(pt, append_to),
            }
        }
    }
}

pub fn one_or_more<P, E, S, F, T>(
    parser: F,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, Vec<T>, E>
where
    F: Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    P: Point,
    E: Recoverable,
{
    move |pm, pt| {
        let (pt, head) = try_parse!(parser(pm, pt));
        let append_to = vec![head];
        let (pt, tail) = try_parse!(zero_or_more_append(append_to, parser)(pm, pt));

        Progress::success(pt, tail)
    }
}

pub fn one_or_more_append<P, E, S, A, F, T>(
    append_to: A,
    parser: F,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, Vec<T>, E>
where
    F: Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    A: IntoAppend<T>,
    P: Point,
    E: Recoverable,
{
    move |pm, pt| {
        let mut append_to = append_to.into();
        let (pt, head) = try_parse!(parser(pm, pt));
        append_to.push(head);
        let (pt, tail) = try_parse!(zero_or_more_append(append_to, parser)(pm, pt));

        Progress::success(pt, tail)
    }
}

pub fn map<P, E, S, F, C, T, U>(
    parser: F,
    convert: C,
) -> impl FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, U, E>
where
    F: FnOnce(&mut ParseMaster<P, E, S>, P) -> Progress<P, T, E>,
    C: FnOnce(T) -> U,
    P: Point,
    E: Recoverable,
{
    move |pm, pt| parser(pm, pt).map(convert)
}

pub fn point<P, E, S>(_: &mut ParseMaster<P, E, S>, pt: P) -> Progress<P, P, E>
where
    P: Clone,
{
    Progress::success(pt.clone(), pt)
}

pub fn inspect<P, E, S, F>(f: F) -> impl Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, (), E>
where
    F: Fn(&P),
{
    move |_, pt| {
        f(&pt);
        Progress::success(pt, ())
    }
}

pub fn state<P, E, S, F>(f: F) -> impl Fn(&mut ParseMaster<P, E, S>, P) -> Progress<P, (), E>
where
    F: Fn(&mut S),
{
    move |pm, pt| {
        f(&mut pm.state);
        Progress::success(pt, ())
    }
}
