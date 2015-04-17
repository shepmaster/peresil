//! # Ingredient list parser
//!
//! ## Example
//!
//! 2 cups fresh snow peas
//! 1 oz water
//! 6 tbsp baking soda
//!
//! ## Grammar
//!
//! Ingredients := Ingredient*
//! Ingredient  := Amount Name "\n"
//! Amount      := Number Unit
//! Number      := [0-9]+
//! Unit        := cups | cup | c
//!             := ounces | ounce | oz
//!             := tablespoons | tablespoon | tbsp
//! Name        := [^\n]*

#[macro_use]
extern crate peresil;

use std::borrow::ToOwned;
use peresil::{ParseMaster, StringPoint, Recoverable};

#[derive(Debug,Copy,Clone,PartialEq)]
enum Unit {
    Cup,
    Ounce,
    Tablespoon
}

#[derive(Debug,Copy,Clone,PartialEq)]
struct Amount {
    unit: Unit,
    size: u8,
}

#[derive(Debug,Clone,PartialEq)]
struct Ingredient {
    amount: Amount,
    name: String,
}

#[derive(Debug,Copy,Clone,PartialEq)]
enum Error {
    ExpectedWhitespace,
    ExpectedNumber,
    UnknownUnit,
    ExpectedName,
    InputRemaining,
}

impl Recoverable for Error {
    fn recoverable(&self) -> bool {
        use Error::*;

        match *self {
            ExpectedWhitespace | ExpectedNumber | ExpectedName => true,
            InputRemaining => true,
            UnknownUnit  => false,
        }
    }
}

type IngredientMaster<'a> = peresil::ParseMaster<StringPoint<'a>, Error>;
type IngredientProgress<'a, T> = peresil::Progress<StringPoint<'a>, T, Error>;

/// Parse a sequence of one-or-more ASCII space characters
fn parse_whitespace<'a>(_: &mut IngredientMaster<'a>, pt: StringPoint<'a>)
                        -> IngredientProgress<'a, &'a str>
{
    let digits = pt.s.chars().take_while(|&c| c == ' ').count();
    let r = if digits == 0 { pt.consume_to(None) } else { pt.consume_to(Some(digits)) };

    r.map_err(|_| Error::ExpectedWhitespace)
}

/// Parse a sequence of one-or-more ASCII digits
fn parse_number<'a>(_: &mut IngredientMaster<'a>, pt: StringPoint<'a>)
                    -> IngredientProgress<'a, u8>
{
    // We can cheat and know that ASCII 0-9 only takes one byte each
    let digits = pt.s.chars().take_while(|&c| c >= '0' && c <= '9').count();
    let r = if digits == 0 { pt.consume_to(None) } else { pt.consume_to(Some(digits)) };

    let (pt, v) = try_parse!(r.map_err(|_| Error::ExpectedNumber));

    let num = v.parse().unwrap();
    peresil::Progress::success(pt, num)
}

fn parse_unit<'a>(_: &mut IngredientMaster<'a>, pt: StringPoint<'a>)
                  -> IngredientProgress<'a, Unit>
{
    let identifiers = &[
        ("cups", Unit::Cup),
        ("cup",  Unit::Cup),
        ("c",    Unit::Cup),
        ("ounces", Unit::Ounce),
        ("ounce",  Unit::Ounce),
        ("oz",     Unit::Ounce),
        ("tablespoons", Unit::Tablespoon),
        ("tablespoon",  Unit::Tablespoon),
        ("tbsp",        Unit::Tablespoon),
    ];

    pt.consume_identifier(identifiers).map_err(|_| Error::UnknownUnit)
}

/// Parse a sequence of 1-or-more characters that aren't newlines
fn parse_name<'a>(_: &mut IngredientMaster<'a>, pt: StringPoint<'a>)
                  -> IngredientProgress<'a, &'a str>
{
    let len = pt.s.len();
    let end_of_name = pt.s.find('\n')
        .or(if len > 0 { Some(len) } else { None });

    pt.consume_to(end_of_name).map_err(|_| Error::ExpectedName)
}

fn parse_ingredient<'a>(pm: &mut IngredientMaster<'a>, pt: StringPoint<'a>)
                        -> IngredientProgress<'a, Ingredient>
{
    let (pt, size) = try_parse!(parse_number(pm, pt));
    let (pt, _)    = try_parse!(parse_whitespace(pm, pt));
    let (pt, unit) = try_parse!(parse_unit(pm, pt));
    let (pt, _)    = try_parse!(parse_whitespace(pm, pt));
    let (pt, name) = try_parse!(parse_name(pm, pt));

    let i = Ingredient { amount: Amount { size: size, unit: unit }, name: name.to_owned() };
    peresil::Progress::success(pt, i)
}

fn parse(s: &str) -> Result<Vec<Ingredient>, (usize, Vec<Error>)> {
    let mut pm = ParseMaster::new();
    let pt = StringPoint::new(s);

    let r = pm.zero_or_more(pt, |pm, pt| parse_ingredient(pm, pt));

    // Check if there's input left
    let r = match r {
        peresil::Progress { status: peresil::Status::Success(..), point: StringPoint { s: "", .. } } => r,
        peresil::Progress { status: peresil::Status::Success(..), point } => {
            peresil::Progress::failure(point, Error::InputRemaining)
        }
        _ => r
    };

    match pm.finish(r) {
        peresil::Progress { status: peresil::Status::Success(v), .. } => Ok(v),
        peresil::Progress { status: peresil::Status::Failure(e), point } => Err((point.offset, e)),
    }
}

#[test]
fn cups() {
    assert_eq!(
        parse("2 cups fresh snow peas"),
        Ok(vec![Ingredient { name: "fresh snow peas".to_owned(),
                             amount: Amount { size: 2, unit: Unit::Cup } }])
    );
}

#[test]
fn ounces() {
    assert_eq!(
        parse("1 oz water"),
        Ok(vec![Ingredient { name: "water".to_owned(),
                             amount: Amount { size: 1, unit: Unit::Ounce } }])
    );
}

#[test]
fn tablespoons() {
    assert_eq!(
        parse("6 tbsp baking soda"),
        Ok(vec![Ingredient { name: "baking soda".to_owned(),
                             amount: Amount { size: 6, unit: Unit::Tablespoon } }])
    );
}

#[test]
fn failure_invalid_size() {
    assert_eq!(
        parse("many tbsp salt"),
        Err((0, vec![Error::ExpectedNumber, Error::InputRemaining]))
    );
}

#[test]
fn failure_expected_whitespace_after_size() {
    assert_eq!(
        parse("5cup"),
        Err((1, vec![Error::ExpectedWhitespace]))
    );
}

#[test]
fn failure_unknown_unit() {
    assert_eq!(
        parse("100 grains rice"),
        Err((4, vec![Error::UnknownUnit]))
    );
}

#[test]
fn failure_expected_whitespace_after_unit() {
    assert_eq!(
        parse("10 cups"),
        Err((7, vec![Error::ExpectedWhitespace]))
    );
}

#[test]
fn failure_expected_name() {
    assert_eq!(
        parse("10 cups "),
        Err((8, vec![Error::ExpectedName]))
    );
}
