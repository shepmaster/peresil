//! A simple and simplistic parsing library
//!
//! ### Example
//!
//! ```
//! #[macro_use]
//! extern crate peresil;
//!
//! use peresil::{ParseMaster,Progress,Recoverable,Status,StringPoint};
//!
//! type DemoMaster<'a> = ParseMaster<StringPoint<'a>, DemoError>;
//! type DemoProgress<'a, T> = Progress<StringPoint<'a>, T, DemoError>;
//! enum DemoError {
//!     ExpectedGreeting,
//!     ExpectedWhitespace,
//!     ExpectedObject,
//! }
//!
//! impl Recoverable for DemoError {
//!     fn recoverable(&self) -> bool { true }
//! }
//!
//! fn parse_basic<'a>(pm: &mut DemoMaster<'a>, pt: StringPoint<'a>)
//!                   -> DemoProgress<'a, (&'a str, &'a str)>
//! {
//!     let tmp = pm.alternate()
//!         .one(|_| pt.consume_literal("goodbye").map_err(|_| DemoError::ExpectedGreeting))
//!         .one(|_| pt.consume_literal("hello").map_err(|_| DemoError::ExpectedGreeting))
//!         .finish();
//!     let (pt, greeting) = try_parse!(tmp);
//!
//!     let (pt, _) = try_parse!(pt.consume_literal(" ").map_err(|_| DemoError::ExpectedWhitespace));
//!
//!     let tmp = pm.alternate()
//!         .one(|_| pt.consume_literal("world").map_err(|_| DemoError::ExpectedObject))
//!         .one(|_| pt.consume_literal("moon").map_err(|_| DemoError::ExpectedObject))
//!         .finish();
//!     let (pt, object) = try_parse!(tmp);
//!
//!     Progress::success(pt, (greeting, object))
//! }
//!
//! fn main() {
//!     let mut pm = ParseMaster::new();
//!     let pt = StringPoint::new("hello world");
//!
//!     let result = parse_basic(&mut pm, pt);
//!     let (greeting, object) = match pm.finish(result) {
//!         Progress { status: Status::Success(v), .. } => v,
//!         Progress { status: Status::Failure(..), .. } => panic!("Did not parse"),
//!     };
//!
//!     println!("Parsed [{}], [{}]", greeting, object);
//! }
//!

/// A location in the parsed data
pub trait Point: Ord + Copy {
    /// The initial point
    fn zero() -> Self;
}

impl Point for usize { fn zero() -> usize { 0 } }
impl Point for i32 { fn zero() -> i32 { 0 } }

/// Indicate if an error should terminate all parsing.
///
/// Non-recoverable errors will not allow for alternatives to be
/// tried, basically unwinding the parsing stack all the way back to
/// the beginning. Unrecoverable errors are useful for errors that
/// indicate that the content was well-formed but not semantically
/// correct.
pub trait Recoverable {
    fn recoverable(&self) -> bool;
}

#[derive(Debug,PartialEq)]
struct Failures<P, E> {
    point: P,
    kinds: Vec<E>,
}

use std::cmp::Ordering;

impl<P, E> Failures<P, E>
    where P: Point,
{
    fn new() -> Failures<P, E> { Failures { point: P::zero(), kinds: Vec::new() } }

    fn add(&mut self, point: P, failure: E) {
        match point.cmp(&self.point) {
            Ordering::Less => {
                // Do nothing, our existing failures are better
            },
            Ordering::Greater => {
                // The new failure is better, toss existing failures
                self.replace(point, failure);
            },
            Ordering::Equal => {
                // Multiple failures at the same point, tell the user all
                // the ways they could do better.
                self.kinds.push(failure);
            },
        }
    }

    fn replace(&mut self, point: P, failure: E) {
        self.point = point;
        self.kinds.clear();
        self.kinds.push(failure);
    }

    fn into_progress<T>(self) -> Progress<P, T, Vec<E>> {
        Progress { point: self.point, status: Status::Failure(self.kinds) }
    }
}

/// An analog to `Result`, specialized for parsing.
#[derive(Debug,PartialEq)]
pub enum Status<T, E> {
    Success(T),
    Failure(E)
}

impl<T, E> Status<T, E> {
    fn map<F, T2>(self, f: F) -> Status<T2, E>
        where F: FnOnce(T) -> T2
    {
        match self {
            Status::Success(x) => Status::Success(f(x)),
            Status::Failure(x) => Status::Failure(x),
        }
    }

    fn map_err<F, E2>(self, f: F) -> Status<T, E2>
        where F: FnOnce(E) -> E2
    {
        match self {
            Status::Success(x) => Status::Success(x),
            Status::Failure(x) => Status::Failure(f(x)),
        }
    }
}

/// Tracks where the parser currently is and if it is successful.
///
/// On success, some value has been parsed. On failure, nothing has
/// been parsed and the value indicates the reason for the failure.
/// The returned point indicates where to next start parsing, often
/// unchanged on failure.
#[must_use]
#[derive(Debug,PartialEq)]
pub struct Progress<P, T, E> {
    /// The current location.
    pub point: P,
    /// If the point indicates the location of a successful or failed parse.
    pub status: Status<T, E>,
}

impl<P, T, E> Progress<P, T, E> {
    pub fn success(point: P, val: T) -> Progress<P, T, E> {
        Progress { point: point, status: Status::Success(val) }
    }

    pub fn failure(point: P, val: E) -> Progress<P, T, E> {
        Progress { point: point, status: Status::Failure(val) }
    }

    /// Convert the success value, if there is one.
    pub fn map<F, T2>(self, f: F) -> Progress<P, T2, E>
        where F: FnOnce(T) -> T2
    {
        Progress { point: self.point, status: self.status.map(f) }
    }

    /// Convert the failure value, if there is one.
    pub fn map_err<F, E2>(self, f: F) -> Progress<P, T, E2>
        where F: FnOnce(E) -> E2
    {
        Progress { point: self.point, status: self.status.map_err(f) }
    }

    /// Returns the value on success, or rewinds the point and returns
    /// `None` on failure.
    pub fn optional(self, reset_to: P) -> (P, Option<T>) {
        // If we fail N optionals and then a required, it'd be nice to
        // report all the optional things. Might be difficult to do
        // that and return the optional value.
        match self {
            Progress { status: Status::Success(val), point } => (point, Some(val)),
            Progress { status: Status::Failure(..), .. } => (reset_to, None),
        }
    }
}

/// The main entrypoint to parsing.
///
/// This tracks the collection of outstanding errors and provides
/// helper methods for parsing alternative paths and sequences of
/// other parsers.
#[derive(Debug,PartialEq)]
pub struct ParseMaster<P, E> {
    failures: Failures<P, E>,
}

impl<'a, P, E> ParseMaster<P, E>
    where P: Point,
          E: Recoverable,
{
    pub fn new() -> ParseMaster<P, E> {
        ParseMaster {
            failures: Failures::new(),
        }
    }

    fn consume<T>(&mut self, progress: Progress<P, T, E>) -> Progress<P, T, ()> {
        match progress {
            Progress { status: Status::Success(..), .. } => progress.map_err(|_| ()),
            Progress { status: Status::Failure(f), point } => {
                if f.recoverable() {
                    self.failures.add(point, f);
                } else {
                    self.failures.replace(point, f);
                }
                Progress { status: Status::Failure(()), point: point }
            }
        }
    }

    /// Returns the value on success, or rewinds the point and returns
    /// `None` on a recoverable failure. Non-recoverable failures are
    /// propagated.
    pub fn optional<T, F>(&mut self, point: P, mut parser: F)
                          -> Progress<P, Option<T>, E>
        where F: FnMut(&mut ParseMaster<P, E>, P) -> Progress<P, T, E>,
    {
        let orig_point = point;
        // If we fail N optionals and then a required, it'd be nice to
        // report all the optional things. Might be difficult to do
        // that and return the optional value.
        match parser(self, point) {
            Progress { status: Status::Success(val), point } => {
                Progress::success(point, Some(val))
            },
            Progress { status: Status::Failure(f), point } => {
                if f.recoverable() {
                    Progress::success(orig_point, None)
                } else {
                    Progress::failure(point, f)
                }
            },
        }
    }

    /// Run sub-parsers in order until one succeeds.
    pub fn alternate<'pm, T>(&'pm mut self) -> Alternate<'pm, P, T, E> {
        Alternate {
            master: self,
            current: None,
        }
    }

    /// Runs the parser until it fails.
    ///
    /// If the parser fails due to a recoverable error, a collection
    /// of values will be returned and the point will be at the end of
    /// the last successful parse.  If the error is not recoverable,
    /// the error will be passed through directly.
    pub fn zero_or_more<F, T>(&mut self, point: P, mut parser: F) -> Progress<P, Vec<T>, E>
        where F: FnMut(&mut ParseMaster<P, E>, P) -> Progress<P, T, E>
    {
        let mut current_point = point;
        let mut values = Vec::new();

        loop {
            let progress = parser(self, current_point);
            match progress {
                Progress { status: Status::Success(v), point } => {
                    values.push(v);
                    current_point = point;
                },
                Progress { status: Status::Failure(f), point } => {
                    if f.recoverable() {
                        self.failures.add(point, f);
                        break;
                    } else {
                        return Progress { status: Status::Failure(f), point: point };
                    }
                },
            }
        }

        Progress { status: Status::Success(values), point: current_point }
    }

    /// When parsing is complete, provide the final result and gain
    /// access to all failures. Will be recycled and may be used for
    /// further parsing.
    pub fn finish<T>(&mut self, progress: Progress<P, T, E>) -> Progress<P, T, Vec<E>> {
        let progress = self.consume(progress);

        match progress {
            Progress { status: Status::Success(..), .. } => progress.map_err(|_| Vec::new()),
            Progress { status: Status::Failure(..), .. } => {
                use std::mem;
                let f = mem::replace(&mut self.failures, Failures::new());
                f.into_progress()
            },
        }
    }
}

/// Follows the first successful parsing path.
#[must_use]
pub struct Alternate<'pm, P : 'pm, T, E : 'pm> {
    master: &'pm mut ParseMaster<P, E>,
    current: Option<Progress<P, T, E>>,
}

impl<'pm, P, T, E> Alternate<'pm, P, T, E>
    where P: Point,
          E: Recoverable,
{
    fn run_one<F>(&mut self, parser: F)
        where F: FnOnce(&mut ParseMaster<P, E>) -> Progress<P, T, E>
    {
        let r = parser(self.master);
        if let Some(prev) = self.current.take() {
            // We don't care about the previous error, once we've consumed it
            let _ = self.master.consume(prev);
        }
        self.current = Some(r);
    }

    /// Run one alternative parser.
    pub fn one<F>(mut self, parser: F) -> Alternate<'pm, P, T, E>
        where F: FnOnce(&mut ParseMaster<P, E>) -> Progress<P, T, E>
    {
        let recoverable =
            if let Some(Progress { status: Status::Failure(ref f), .. }) = self.current {
                f.recoverable()
            } else {
                false
            };

        match self.current {
            None => self.run_one(parser),
            Some(Progress { status: Status::Success(..), .. }) => {},
            Some(Progress { status: Status::Failure(..), .. })
                if recoverable => self.run_one(parser),
            Some(Progress { status: Status::Failure(..), .. }) => {},
        }

        self
    }

    /// Complete the alternatives, returning the first successful branch.
    pub fn finish(self) -> Progress<P, T, E> {
        self.current.unwrap()
    }
}

/// An analog to `try!`, but for `Progress`
#[macro_export]
macro_rules! try_parse(
    ($e:expr) => ({
        match $e {
            $crate::Progress { status: $crate::Status::Success(val), point } => (point, val),
            $crate::Progress { status: $crate::Status::Failure(val), point } => {
                return $crate::Progress { point: point, status: $crate::Status::Failure(val) }
            }
        }
    });
);

/// Matches a literal string to a specific type, usually an enum.
pub type Identifier<'a, T> = (&'a str, T);

/// Tracks the location of parsing in a string, the most common case.
///
/// Helper methods are provided to do basic parsing tasks, such as
/// finding literal strings.
#[derive(Debug,Copy,Clone,PartialEq,Eq)]
pub struct StringPoint<'a> {
    /// The portion of the input string to start parsing next
    pub s: &'a str,
    /// How far into the original string we are
    pub offset: usize,
}

impl<'a> PartialOrd for StringPoint<'a> {
    #[inline]
    fn partial_cmp(&self, other: &StringPoint<'a>) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<'a> Ord for StringPoint<'a> {
    #[inline]
    fn cmp(&self, other: &StringPoint<'a>) -> Ordering {
        self.offset.cmp(&other.offset)
    }
}

impl<'a> Point for StringPoint<'a> {
    fn zero() -> StringPoint<'a> { StringPoint { s: "", offset: 0} }
}

impl<'a> StringPoint<'a> {
    #[inline]
    pub fn new(s: &'a str) -> StringPoint<'a> {
        StringPoint { s: s, offset: 0 }
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.s.is_empty()
    }

    /// Slices the string.
    #[inline]
    pub fn to(self, other: StringPoint<'a>) -> &'a str {
        let len = other.offset - self.offset;
        &self.s[..len]
    }

    #[inline]
    fn success(self, len: usize) -> Progress<StringPoint<'a>, &'a str, ()> {
        let matched = &self.s[..len];
        let rest = &self.s[len..];

        Progress {
            point: StringPoint { s: rest, offset: self.offset + len },
            status: Status::Success(matched)
        }
    }

    #[inline]
    fn fail<T>(self) -> Progress<StringPoint<'a>, T, ()> {
        Progress { point: self, status: Status::Failure(()) }
    }

    /// Advances the point by the number of bytes. If the value is
    /// `None`, then no value was able to be consumed, and the result
    /// is a failure.
    #[inline]
    pub fn consume_to(&self, l: Option<usize>) -> Progress<StringPoint<'a>, &'a str, ()> {
        match l {
            None => self.fail(),
            Some(position) => self.success(position),
        }
    }

    /// Advances the point if it starts with the literal.
    #[inline]
    pub fn consume_literal(self, val: &str) -> Progress<StringPoint<'a>, &'a str, ()> {
        if self.s.starts_with(val) {
            self.success(val.len())
        } else {
            self.fail()
        }
    }

    /// Iterates through the identifiers and advances the point on the
    /// first matching identifier.
    #[inline]
    pub fn consume_identifier<T>(self, identifiers: &[Identifier<T>])
                                 -> Progress<StringPoint<'a>, T, ()>
        where T: Clone
    {
        for &(identifier, ref item) in identifiers {
            if self.s.starts_with(identifier) {
                return self
                    .consume_to(Some(identifier.len()))
                    .map(|_| item.clone())
                    .map_err(|_| unreachable!());
            }
        }

        self.fail()
    }
}

#[cfg(test)]
mod test {
    use super::{ParseMaster,Progress,Status,StringPoint,Recoverable};

    #[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
    struct AnError(u8);

    impl Recoverable for AnError {
        fn recoverable(&self) -> bool { self.0 < 0x80 }
    }

    type SimpleMaster = ParseMaster<usize, AnError>;
    type SimpleProgress<T> = Progress<usize, T, AnError>;

    #[test]
    fn one_error() {
        let mut d = ParseMaster::new();

        let r = d.finish::<()>(Progress { point: 0, status: Status::Failure(AnError(1)) });

        assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(1)]) });
    }

    #[test]
    fn two_error_at_same_point() {
        let mut d = ParseMaster::new();

        let r = d.alternate::<()>()
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(2)) })
            .finish();

        let r = d.finish(r);

        assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(1), AnError(2)]) });
    }

    #[test]
    fn first_error_is_better() {
        let mut d = ParseMaster::new();

        let r = d.alternate::<()>()
            .one(|_| Progress { point: 1, status: Status::Failure(AnError(1)) })
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(2)) })
            .finish();

        let r = d.finish(r);

        assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(1)]) });
    }

    #[test]
    fn second_error_is_better() {
        let mut d = ParseMaster::new();

        let r = d.alternate::<()>()
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
            .one(|_| Progress { point: 1, status: Status::Failure(AnError(2)) })
            .finish();

        let r = d.finish(r);

        assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(2)]) });
    }

    #[test]
    fn one_success() {
        let mut d = ParseMaster::<_, AnError>::new();

        let r = d.finish(Progress { point: 0, status: Status::Success(42) });

        assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
    }

    #[test]
    fn success_after_failure() {
        let mut d = ParseMaster::new();

        let r = d.alternate()
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(1)) })
            .one(|_| Progress { point: 0, status: Status::Success(42) })
            .finish();

        let r = d.finish(r);
        assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
    }

    #[test]
    fn success_before_failure() {
        let mut d = ParseMaster::<_, AnError>::new();

        let r = d.alternate()
            .one(|_| Progress { point: 0, status: Status::Success(42) })
            .one(|_| panic!("Should not even be called"))
            .finish();

        let r = d.finish(r);
        assert_eq!(r, Progress { point: 0, status: Status::Success(42) });
    }

    #[test]
    fn sequential_success() {
        fn first(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(1) }
        }

        fn second(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(2) }
        }

        fn both(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
            let (pt, val1) = try_parse!(first(d, pt));
            let (pt, val2) = try_parse!(second(d, pt));
            Progress { point: pt, status: Status::Success((val1, val2)) }
        }

        let mut d = ParseMaster::new();
        let r = both(&mut d, 0);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 2, status: Status::Success((1,2)) });
    }

    #[test]
    fn child_parse_succeeds() {
        fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
            let (pt, val1) = try_parse!(child(d, pt));
            Progress { point: pt + 1, status: Status::Success((val1, 2)) }
        }

        fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(1) }
        }

        let mut d = ParseMaster::new();
        let r = parent(&mut d, 0);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 2, status: Status::Success((1, 2)) });
    }

    #[test]
    fn child_parse_fails_child_step() {
        fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
            let (pt, val1) = try_parse!(child(d, pt));
            Progress { point: pt + 1, status: Status::Success((val1, 2)) }
        }

        fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }

        let mut d = ParseMaster::new();
        let r = parent(&mut d, 0);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(1)]) });
    }

    #[test]
    fn child_parse_fails_parent_step() {
        fn parent(d: &mut SimpleMaster, pt: usize) -> SimpleProgress<(u8,u8)> {
            let (pt, _) = try_parse!(child(d, pt));
            Progress { point: pt + 1, status: Status::Failure(AnError(2)) }
        }

        fn child(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(1) }
        }

        let mut d = ParseMaster::new();
        let r = parent(&mut d, 0);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 2, status: Status::Failure(vec![AnError(2)]) });
    }

    #[test]
    fn alternate_with_children_parses() {
        fn first(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }

        fn second(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(1) }
        }

        fn both(d: &mut SimpleMaster, pt: usize) -> Progress<usize, u8, AnError> {
            d.alternate()
                .one(|d| first(d, pt))
                .one(|d| second(d, pt))
                .finish()
        }

        let mut d = ParseMaster::new();
        let r = both(&mut d, 0);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 1, status: Status::Success(1) });
    }

    #[test]
    fn alternate_stops_parsing_after_unrecoverable_failure() {
        let mut d = ParseMaster::new();
        let r = d.alternate()
            .one(|_| Progress { point: 0, status: Status::Failure(AnError(255)) })
            .one(|_| Progress { point: 0, status: Status::Success(()) })
            .finish();
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(255)]) });
    }

    #[test]
    fn optional_present() {
        fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Success(1) }
        }

        let mut d = ParseMaster::new();
        let (pt, val) = optional(&mut d, 0).optional(0);

        assert_eq!(pt, 1);
        assert_eq!(val, Some(1));
    }

    #[test]
    fn optional_missing() {
        fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }

        let mut d = ParseMaster::new();
        let (pt, val) = optional(&mut d, 0).optional(0);

        assert_eq!(pt, 0);
        assert_eq!(val, None);
    }

    #[test]
    fn optional_with_recoverable() {
        fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }

        let mut d = ParseMaster::new();
        let r = d.optional(0, |pm, pt| optional(pm, pt));
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 0, status: Status::Success(None) });
    }

    #[test]
    fn optional_with_unrecoverable() {
        fn optional(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(255)) }
        }

        let mut d = ParseMaster::new();
        let r = d.optional(0, |pm, pt| optional(pm, pt));
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 1, status: Status::Failure(vec![AnError(255)]) });
    }

    #[test]
    fn zero_or_more() {
        let mut remaining: u8 = 2;

        let mut body = |_: &mut SimpleMaster, pt: usize| -> SimpleProgress<u8> {
            if remaining > 0 {
                remaining -= 1;
                Progress { point: pt + 1, status: Status::Success(remaining) }
            } else {
                Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
            }
        };

        let mut d = ParseMaster::new();
        let r = d.zero_or_more(0, |d, pt| body(d, pt));
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 2, status: Status::Success(vec![1, 0]) });
    }

    #[test]
    fn zero_or_more_failure_returns_to_beginning_of_line() {
        fn body(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt + 1, status: Status::Failure(AnError(1)) }
        }

        let mut d = ParseMaster::new();
        let r = d.zero_or_more(0, |d, pt| body(d, pt));
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 0, status: Status::Success(vec![]) });
    }

    #[test]
    fn zero_or_more_fails_on_unrecoverable_failure() {
        fn body(_: &mut SimpleMaster, pt: usize) -> SimpleProgress<u8> {
            Progress { point: pt, status: Status::Failure(AnError(255)) }
        }

        let mut d = ParseMaster::new();
        let r = d.zero_or_more(0, |d, pt| body(d, pt));
        let r = d.finish(r);

        assert_eq!(r, Progress { point: 0, status: Status::Failure(vec![AnError(255)]) });
    }

    type StringMaster<'a> = ParseMaster<StringPoint<'a>, AnError>;
    type StringProgress<'a, T> = Progress<StringPoint<'a>, T, AnError>;

    #[test]
    fn string_sequential() {
        fn all<'a>(pt: StringPoint<'a>) -> StringProgress<'a, (&'a str, &'a str, &'a str)> {
            let (pt, a) = try_parse!(pt.consume_literal("a").map_err(|_| AnError(1)));
            let (pt, b) = try_parse!(pt.consume_literal("b").map_err(|_| AnError(2)));
            let (pt, c) = try_parse!(pt.consume_literal("c").map_err(|_| AnError(3)));

            Progress { point: pt, status: Status::Success((a,b,c)) }
        }

        let mut d = ParseMaster::new();
        let pt = StringPoint::new("abc");

        let r = all(pt);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: StringPoint { s: "", offset: 3 }, status: Status::Success(("a", "b", "c")) });
    }

    #[test]
    fn string_alternate() {
        fn any<'a>(d: &mut StringMaster<'a>, pt: StringPoint<'a>) -> StringProgress<'a, &'a str> {
            d.alternate()
                .one(|_| pt.consume_literal("a").map_err(|_| AnError(1)))
                .one(|_| pt.consume_literal("b").map_err(|_| AnError(2)))
                .one(|_| pt.consume_literal("c").map_err(|_| AnError(3)))
                .finish()
        }

        let mut d = ParseMaster::new();
        let pt = StringPoint::new("c");

        let r = any(&mut d, pt);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: StringPoint { s: "", offset: 1 }, status: Status::Success("c") });
    }

    #[test]
    fn string_zero_or_more() {
        fn any<'a>(d: &mut StringMaster<'a>, pt: StringPoint<'a>) -> StringProgress<'a, Vec<&'a str>> {
            d.zero_or_more(pt, |_, pt| pt.consume_literal("a").map_err(|_| AnError(1)))
        }

        let mut d = ParseMaster::new();
        let pt = StringPoint::new("aaa");

        let r = any(&mut d, pt);
        let r = d.finish(r);

        assert_eq!(r, Progress { point: StringPoint { s: "", offset: 3 }, status: Status::Success(vec!["a", "a", "a"]) });
    }

    #[test]
    fn string_to() {
        let pt1 = StringPoint::new("hello world");
        let pt2 = StringPoint { offset: pt1.offset + 5, s: &pt1.s[5..] };
        assert_eq!("hello", pt1.to(pt2));
    }

    #[test]
    fn string_consume_literal() {
        let pt = StringPoint::new("hello world");

        let r = pt.consume_literal("hello");
        assert_eq!(r, Progress { point: StringPoint { s: " world", offset: 5 },
                                 status: Status::Success("hello") });

        let r = pt.consume_literal("goodbye");
        assert_eq!(r, Progress { point: StringPoint { s: "hello world", offset: 0 },
                                 status: Status::Failure(()) });
    }

    #[test]
    fn string_consume_identifier() {
        let pt = StringPoint::new("hello world");

        let r = pt.consume_identifier(&[("goodbye", 1), ("hello", 2)]);
        assert_eq!(r, Progress { point: StringPoint { s: " world", offset: 5 },
                                 status: Status::Success(2) });

        let r = pt.consume_identifier(&[("red", 3), ("blue", 4)]);
        assert_eq!(r, Progress { point: StringPoint { s: "hello world", offset: 0 },
                                 status: Status::Failure(()) });
    }
}
