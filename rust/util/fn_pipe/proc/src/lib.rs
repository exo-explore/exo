//! Proc-macro for implementing `Fn/Pipe*` variants for tuples of a given size;
//! it is only here for this one purpose and no other, should not be used elsewhere

#![allow(clippy::arbitrary_source_item_ordering)]

extern crate proc_macro;

use extend::ext;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitInt};

type TokS2 = proc_macro2::TokenStream;

#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::missing_panics_doc,
    clippy::too_many_lines
)]
#[proc_macro]
pub fn impl_fn_pipe_for_tuple(item: TokenStream) -> TokenStream {
    // DEFINE CONSTANT TOKEN STREAMS UPFRONT
    // token streams for Fn/Pipe* variants
    let fn_pipe_names = (
        (
            "Fn".parse_unchecked(),
            "FnPipe".parse_unchecked(),
            "run".parse_unchecked(),
            "call".parse_unchecked(),
        ),
        (
            "FnMut".parse_unchecked(),
            "FnMutPipe".parse_unchecked(),
            "run_mut".parse_unchecked(),
            "call_mut".parse_unchecked(),
        ),
        (
            "FnOnce".parse_unchecked(),
            "FnOncePipe".parse_unchecked(),
            "run_once".parse_unchecked(),
            "call_once".parse_unchecked(),
        ),
    );

    // get the number of tuple parameters to implement this for
    let max_tuple_size = match parse_macro_input!(item as LitInt).base10_parse::<usize>() {
        Ok(num) => num,
        Err(e) => return e.to_compile_error().into(),
    };
    assert!(
        max_tuple_size > 0,
        "passed parameter must be greater than zero"
    );

    // generate generic function type-names, to be used later everywhere
    let mut fn_type_names = Vec::with_capacity(max_tuple_size);
    for i in 0..max_tuple_size {
        fn_type_names.push(format!("_{i}").parse_unchecked());
    }

    // create a middle type constraint (i.e. not the first one)
    let middle_type_constraint = |prev_fn: TokS2, this_fn: TokS2, fn_name: TokS2| {
        quote! {
            #this_fn: #fn_name<(#prev_fn::Output,)>
        }
    };

    // create call implementation
    let impl_call = |n: usize, call: TokS2, base: TokS2| {
        let tuple_access = format!("self.{n}").parse_unchecked();
        quote! {
            #tuple_access.#call((#base,))
        }
    };

    // generic impl block parametrised on the variant and number of params
    let impl_per_type_and_n = |n: usize,
                               (fn_name, fn_pipe_name, run, call): (TokS2, TokS2, TokS2, TokS2),
                               extra: Option<TokS2>,
                               ref_style: Option<TokS2>| {
        // flatten the extra tokens
        let extra = extra.unwrap_or_default();

        let fn_type_names_comma_sep = &fn_type_names[0..n].comma_separated();

        // get name of first type and create the type constraint for the fist type
        let first_fn_type = fn_type_names[0].clone();
        let first_type_constraint = quote! {
            #first_fn_type: #fn_name<Args>
        };

        // create the middle type constraint implementations
        let middle_type_constraints = (1..n)
            .map(|i| {
                // get previous and current tokens
                let prev_fn = fn_type_names[i - 1].clone();
                let this_fn = fn_type_names[i].clone();

                // create middle implementation
                middle_type_constraint(prev_fn, this_fn, fn_name.clone())
            })
            .collect::<Vec<_>>();

        // combine the two, and comma-separate them into a single block
        let type_constraints = [vec![first_type_constraint], middle_type_constraints]
            .concat()
            .as_slice()
            .comma_separated();

        // recursive call implementation starting from the base
        let mut call_impl = quote! { self.0 .#call(args) };
        for i in 1..n {
            call_impl = impl_call(i, call.clone(), call_impl);
        }

        quote! {
            #[allow(clippy::type_repetition_in_bounds)]
            impl<Args: Tuple, #fn_type_names_comma_sep: ?Sized> #fn_pipe_name<Args> for (#fn_type_names_comma_sep,)
            where #type_constraints
            {
                #extra

                #[inline]
                extern "rust-call" fn #run(#ref_style self, args: Args) -> Self::Output {
                    #call_impl
                }
            }
        }
    };

    // generic impl block parametrised on the number of params
    let impl_per_n = |n: usize| {
        // create the `Fn/FnPipe` implementation
        let mut impl_per_n =
            impl_per_type_and_n(n, fn_pipe_names.0.clone(), None, Some(quote! { & }));

        // create the `FnMut/FnMutPipe` implementation
        impl_per_n.extend(impl_per_type_and_n(
            n,
            fn_pipe_names.1.clone(),
            None,
            Some(quote! { &mut }),
        ));

        // create the `FnOnce/FnOncePipe` implementation;
        // this implementation additionally needs to specify the associated `type Output`
        let last = fn_type_names[n - 1].clone();
        impl_per_n.extend(impl_per_type_and_n(
            n,
            fn_pipe_names.2.clone(),
            Some(quote! {
                type Output = #last::Output;
            }),
            None,
        ));

        impl_per_n
    };

    // we need to implement for all tuple sizes 1 through-to `n`
    let mut impls = TokS2::new();
    for n in 1..=max_tuple_size {
        impls.extend(impl_per_n(n));
    }

    // return all the impls
    impls.into()
}

#[ext]
impl [TokS2] {
    #[allow(clippy::unwrap_used, clippy::single_call_fn)]
    fn comma_separated(&self) -> TokS2 {
        let comma_tok = ",".parse_unchecked();

        // get the first token, and turn it into an accumulator
        let mut toks = self.iter();
        let mut tok: TokS2 = toks.next().unwrap().clone();

        // if there are more tokens to come, keep extending with comma
        for next in toks {
            tok.extend(comma_tok.clone());
            tok.extend(next.clone());
        }

        // return final comma-separated result
        tok
    }
}

#[ext]
impl str {
    fn parse_unchecked(&self) -> TokS2 {
        match self.parse::<TokS2>() {
            Ok(s) => s,
            Err(e) => unimplemented!("{e}"),
        }
    }
}
