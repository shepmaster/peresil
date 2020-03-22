//! # Simple 4-function calculator
//!
//! This demonstrates usage of Peresil, and also shows how you can
//! write a recursive-descent parser that handles left-associative
//! operators.
//!
//! For an extra wrinkle, input numbers must be integers in the range
//! [0, 31]. This allows an opportunity to show how errors outside the
//! grammar can be generated at parsing time.
//!
//! ## Grammar
//!
//! Expr := Add
//! Add  := Add '+' Mul
//!      := Add '-' Mul
//!      := Mul
//! Mul  := Mul '*' Num
//!      := Mul '/' Num
//!      := Num
//! Num  := [0-9]+

#[macro_use]
extern crate peresil;

use peresil::{ParseMaster, Recoverable, StringPoint};

// It's recommended to make type aliases to clean up signatures
type CalcMaster<'a> = ParseMaster<StringPoint<'a>, Error>;
type CalcProgress<'a, T> = peresil::Progress<StringPoint<'a>, T, Error>;

#[derive(Debug, Clone, PartialEq)]
enum Expression {
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Num(u8),
}

impl Expression {
    fn evaluate(&self) -> i32 {
        use Expression::*;

        match *self {
            Add(ref l, ref r) => l.evaluate() + r.evaluate(),
            Sub(ref l, ref r) => l.evaluate() - r.evaluate(),
            Mul(ref l, ref r) => l.evaluate() * r.evaluate(),
            Div(ref l, ref r) => l.evaluate() / r.evaluate(),
            Num(v) => v as i32,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Error {
    ExpectedNumber,
    InvalidNumber(u8),
}

impl Recoverable for Error {
    fn recoverable(&self) -> bool {
        use Error::*;

        match *self {
            ExpectedNumber => true,
            InvalidNumber(..) => false,
        }
    }
}

/// Maps an operator to a function that builds the corresponding Expression
type LeftAssociativeRule<'a> = (
    &'static str,
    &'a dyn Fn(Expression, Expression) -> Expression,
);

/// Iteratively parses left-associative operators, avoiding infinite
/// recursion. Provide a `child_parser` that corresponds to each side
/// of the operator, as well as a rule for each operator.
fn parse_left_associative_operator<'a, P>(
    pm: &mut CalcMaster<'a>,
    pt: StringPoint<'a>,
    child_parser: P,
    rules: &[LeftAssociativeRule],
) -> CalcProgress<'a, Expression>
where
    P: for<'b> Fn(&mut CalcMaster<'b>, StringPoint<'b>) -> CalcProgress<'b, Expression>,
{
    let (pt, mut a) = try_parse!(child_parser(pm, pt));
    let mut start = pt;

    loop {
        let mut matched = false;

        for &(ref operator, ref builder) in rules {
            let (pt, op) = start.consume_literal(operator).optional(start);
            if op.is_none() {
                continue;
            }

            let (pt, b) = try_parse!(child_parser(pm, pt));

            a = builder(a, b);
            start = pt;

            matched = true;
            break;
        }

        if !matched {
            break;
        }
    }

    peresil::Progress::success(pt, a)
}

/// Parse a sequence of one-or-more ASCII digits
fn parse_num<'a>(_: &mut CalcMaster<'a>, pt: StringPoint<'a>) -> CalcProgress<'a, Expression> {
    let original_pt = pt;

    // We can cheat and know that ASCII 0-9 only takes one byte each
    let digits = pt.s.chars().take_while(|&c| c >= '0' && c <= '9').count();
    let r = if digits == 0 {
        pt.consume_to(None)
    } else {
        pt.consume_to(Some(digits))
    };

    let (pt, v) = try_parse!(r.map_err(|_| Error::ExpectedNumber));

    let num = v.parse().unwrap();

    // Here's where we can raise our own parsing errors. Note that we
    // kept the point where the number started, in order to give an
    // accurate error position.
    if num > 31 {
        peresil::Progress::failure(original_pt, Error::InvalidNumber(num))
    } else {
        peresil::Progress::success(pt, Expression::Num(num))
    }
}

fn parse_muldiv<'a>(pm: &mut CalcMaster<'a>, pt: StringPoint<'a>) -> CalcProgress<'a, Expression> {
    parse_left_associative_operator(
        pm,
        pt,
        parse_num,
        &[
            ("*", &|a, b| Expression::Mul(Box::new(a), Box::new(b))),
            ("/", &|a, b| Expression::Div(Box::new(a), Box::new(b))),
        ],
    )
}

fn parse_addsub<'a>(pm: &mut CalcMaster<'a>, pt: StringPoint<'a>) -> CalcProgress<'a, Expression> {
    parse_left_associative_operator(
        pm,
        pt,
        parse_muldiv,
        &[
            ("+", &|a, b| Expression::Add(Box::new(a), Box::new(b))),
            ("-", &|a, b| Expression::Sub(Box::new(a), Box::new(b))),
        ],
    )
}

fn parse(s: &str) -> Result<Expression, (usize, Vec<Error>)> {
    let mut pm = ParseMaster::new();
    let pt = StringPoint::new(s);

    let result = parse_addsub(&mut pm, pt);
    match pm.finish(result) {
        peresil::Progress {
            status: peresil::Status::Success(v),
            ..
        } => Ok(v),
        peresil::Progress {
            status: peresil::Status::Failure(f),
            point,
        } => Err((point.offset, f)),
    }
}

fn n(n: u8) -> Box<Expression> {
    Box::new(Expression::Num(n))
}

#[test]
fn single_number() {
    use Expression::*;
    assert_eq!(parse("1"), Ok(Num(1)));
}

#[test]
fn add_two_numbers() {
    use Expression::*;
    assert_eq!(parse("1+2"), Ok(Add(n(1), n(2))));
}

#[test]
fn add_three_numbers() {
    use Expression::*;
    assert_eq!(parse("3+4+5"), Ok(Add(Box::new(Add(n(3), n(4))), n(5))));
}

#[test]
fn subtract_two_numbers() {
    use Expression::*;
    assert_eq!(parse("9-8"), Ok(Sub(n(9), n(8))));
}

#[test]
fn multiply_two_numbers() {
    use Expression::*;
    assert_eq!(parse("5*6"), Ok(Mul(n(5), n(6))));
}

#[test]
fn multiply_three_numbers() {
    use Expression::*;
    assert_eq!(parse("3*6*9"), Ok(Mul(Box::new(Mul(n(3), n(6))), n(9))));
}

#[test]
fn divide_two_numbers() {
    use Expression::*;
    assert_eq!(parse("9/3"), Ok(Div(n(9), n(3))));
}

#[test]
fn addition_adds() {
    assert_eq!(parse("1+2+3").unwrap().evaluate(), 6);
}

#[test]
fn subtraction_subtracts() {
    assert_eq!(parse("1-2-3").unwrap().evaluate(), -4);
}

#[test]
fn multiplication_multiplies() {
    assert_eq!(parse("2*3*4").unwrap().evaluate(), 24);
}

#[test]
fn division_divides() {
    assert_eq!(parse("9/3/3").unwrap().evaluate(), 1);
}

#[test]
fn all_operators_together() {
    assert_eq!(parse("3+2-2*9/3").unwrap().evaluate(), -1);
}

#[test]
fn failure_not_a_number() {
    assert_eq!(parse("cow"), Err((0, vec![Error::ExpectedNumber])));
}

#[test]
fn failure_invalid_number() {
    assert_eq!(parse("32"), Err((0, vec![Error::InvalidNumber(32)])));
}

#[test]
fn failure_invalid_number_in_other_position() {
    assert_eq!(parse("1+99"), Err((2, vec![Error::InvalidNumber(99)])));
}
