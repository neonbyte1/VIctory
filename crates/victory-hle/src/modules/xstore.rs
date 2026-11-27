//! XStore as an offline console answers it: the game's own licence is
//! local and active (a full, owned game); the user owns no add-ons; what
//! needs the store service (catalog, consumables, purchase and update
//! pages, collection ids, update checks) fails with the documented store
//! network error.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};

use super::xasync::{self, AsyncBlock, E_ILLEGAL_METHOD_CALL, E_INVALIDARG};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, put_string, text, token,
};
use super::xpackage;
use crate::runtime;

pub const PROVIDER: Guid = Guid::new(
    0x0dd1_12ac,
    0x7c24,
    0x448c,
    [0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c],
);
/// As Grand Theft Auto VI (2026) asks for it; every slot it calls takes
/// the public API's arguments in the 2020 order.
pub const IID_2026: Guid = Guid::new(
    0xb09d_803c,
    0x2414,
    0x4a05,
    [0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44],
);
/// The same table extended by durables licensing (0x238, 0x240).
pub const DURABLES_IID: Guid = Guid::new(
    0x2d42_fea5,
    0xe71d,
    0x4b76,
    [0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d],
);

pub static OBJECT: Object = Object::new("store", &VTABLE);
pub const INTERFACES: &[Interface] = &[
    (PROVIDER, IID_2026, &OBJECT),
    (PROVIDER, DURABLES_IID, &OBJECT),
];

/// The documented "failed to communicate with the store network".
pub const NETWORK_ERROR: HRESULT = 0x8924_5301_u32 as i32;

type This = *const Object;
type Handle = *const c_void;

/// Contexts and product queries hold nothing (queries are always empty),
/// so their handles are fixed addresses. A licence handle says whether it
/// is valid by which address it is.
static CONTEXT: u8 = 0;
static QUERY: u8 = 0;
static VALID: u8 = 0;
static INVALID: u8 = 0;
fn handle(of: &'static u8) -> Handle {
    (of as *const u8).cast()
}

/// Identifies our calls on a block: one per result shape.
static FAILED: u8 = 0;
static QUERY_CALL: u8 = 0;
static LICENSE_CALL: u8 = 0;
static GAME_LICENSE: u8 = 0;

fn offline(block: *mut AsyncBlock, what: &str) -> HRESULT {
    tracing::info!(what, "store call: offline");
    xasync::finished(block, (&raw const FAILED).cast(), NETWORK_ERROR)
}
/// The result functions of calls that failed: their status.
extern "system" fn status(_: This, block: *mut AsyncBlock) -> HRESULT {
    xasync::XAsyncGetStatus(block, false)
}

extern "system" fn create_context(_: This, _user: *const c_void, out: *mut Handle) -> HRESULT {
    put(out, handle(&CONTEXT))
}
extern "system" fn close(_: This, _: Handle) {}

extern "system" fn associated_products_async(
    _: This,
    _: Handle,
    _kinds: u32,
    _per_page: u32,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "associated products")
}

/// The user owns no add-ons: an empty query.
extern "system" fn entitled_products_async(
    _: This,
    _: Handle,
    _kinds: u32,
    _per_page: u32,
    block: *mut AsyncBlock,
) -> HRESULT {
    xasync::finished_with(
        block,
        (&raw const QUERY_CALL).cast(),
        Ok(xasync::bytes(handle(&QUERY))),
    )
}
extern "system" fn query_result(_: This, block: *mut AsyncBlock, out: *mut Handle) -> HRESULT {
    xasync::result(block, (&raw const QUERY_CALL).cast(), out)
}
/// Empty, so the callback never runs.
extern "system" fn enumerate_products(
    _: This,
    _: Handle,
    _: *mut c_void,
    _: *const c_void,
) -> HRESULT {
    S_OK
}
extern "system" fn has_more_pages(_: This, _: Handle) -> bool {
    false
}
/// There is no next page to ask for.
extern "system" fn next_page_async(_: This, _: Handle, block: *mut AsyncBlock) -> HRESULT {
    xasync::finished(block, (&raw const QUERY_CALL).cast(), E_ILLEGAL_METHOD_CALL)
}

fn license_for(owned: bool, block: *mut AsyncBlock) -> HRESULT {
    let license = handle(if owned { &VALID } else { &INVALID });
    xasync::finished_with(
        block,
        (&raw const LICENSE_CALL).cast(),
        Ok(xasync::bytes(license)),
    )
}
/// Only the game's own package is installed, so only its licence is valid.
extern "system" fn license_for_package_async(
    _: This,
    _: Handle,
    package: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    let ours = text(package).is_some_and(|p| p == xpackage::identifier());
    license_for(ours, block)
}
/// The user owns no durables: their licences are not valid.
extern "system" fn license_for_durables_async(
    _: This,
    _: Handle,
    store_id: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    tracing::debug!(
        store_id = text(store_id).unwrap_or_default(),
        "durable not owned"
    );
    license_for(false, block)
}
extern "system" fn license_result(_: This, block: *mut AsyncBlock, out: *mut Handle) -> HRESULT {
    xasync::result(block, (&raw const LICENSE_CALL).cast(), out)
}
extern "system" fn is_license_valid(_: This, license: Handle) -> bool {
    license == handle(&VALID)
}

/// The game licence, as the public API lays it out.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GameLicense {
    pub sku_store_id: [u8; 18],
    pub is_active: bool,
    pub is_trial_owned_by_this_user: bool,
    pub is_disc_license: bool,
    pub is_trial: bool,
    pub trial_seconds_remaining: u32,
    pub trial_unique_id: [u8; 64],
    pub expiration: i64,
}

impl xasync::ResultBytes for GameLicense {
    fn into_result_bytes(self) -> Vec<u8> {
        let mut bytes = vec![0; size_of::<Self>()];
        let mut field = |offset: usize, value: &[u8]| {
            bytes[offset..offset + value.len()].copy_from_slice(value);
        };
        // Moving even a zero-initialized Rust struct need not preserve its
        // padding. Write the fields into the final byte buffer instead.
        field(std::mem::offset_of!(Self, sku_store_id), &self.sku_store_id);
        field(
            std::mem::offset_of!(Self, is_active),
            &[u8::from(self.is_active)],
        );
        field(
            std::mem::offset_of!(Self, is_trial_owned_by_this_user),
            &[u8::from(self.is_trial_owned_by_this_user)],
        );
        field(
            std::mem::offset_of!(Self, is_disc_license),
            &[u8::from(self.is_disc_license)],
        );
        field(
            std::mem::offset_of!(Self, is_trial),
            &[u8::from(self.is_trial)],
        );
        field(
            std::mem::offset_of!(Self, trial_seconds_remaining),
            &self.trial_seconds_remaining.to_ne_bytes(),
        );
        field(
            std::mem::offset_of!(Self, trial_unique_id),
            &self.trial_unique_id,
        );
        field(
            std::mem::offset_of!(Self, expiration),
            &self.expiration.to_ne_bytes(),
        );
        bytes
    }
}

/// A full, active licence for the game's SKU (`<store id>/0010`, the
/// store's first SKU), not a trial, without expiry.
fn game_license() -> GameLicense {
    let mut sku = [0u8; 18];
    let id = format!("{}/0010", runtime::title().store_id);
    let len = id.len().min(sku.len() - 1);
    sku[..len].copy_from_slice(&id.as_bytes()[..len]);
    GameLicense {
        sku_store_id: sku,
        is_active: true,
        is_trial_owned_by_this_user: false,
        is_disc_license: false,
        is_trial: false,
        trial_seconds_remaining: 0,
        trial_unique_id: [0; 64],
        expiration: 0,
    }
}

extern "system" fn game_license_async(_: This, _: Handle, block: *mut AsyncBlock) -> HRESULT {
    let license = xasync::bytes(game_license());
    xasync::finished_with(block, (&raw const GAME_LICENSE).cast(), Ok(license))
}
extern "system" fn game_license_result(
    _: This,
    block: *mut AsyncBlock,
    out: *mut GameLicense,
) -> HRESULT {
    xasync::result(block, (&raw const GAME_LICENSE).cast(), out)
}

extern "system" fn consumable_balance_async(
    _: This,
    _: Handle,
    _store_id: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "consumable balance")
}
extern "system" fn report_fulfillment_async(
    _: This,
    _: Handle,
    _store_id: *const c_char,
    _quantity: u32,
    _tracking_id: *const c_void,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "consumable fulfillment")
}
/// Collections and purchase ids alike.
extern "system" fn user_id_async(
    _: This,
    _: Handle,
    _ticket: *const c_char,
    _publisher_user_id: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "user store id")
}
extern "system" fn size_status(_: This, block: *mut AsyncBlock, _out: *mut usize) -> HRESULT {
    xasync::XAsyncGetStatus(block, false)
}
extern "system" fn string_status(
    _: This,
    block: *mut AsyncBlock,
    _: usize,
    _: *mut c_char,
) -> HRESULT {
    xasync::XAsyncGetStatus(block, false)
}
extern "system" fn count_status(_: This, block: *mut AsyncBlock, _out: *mut u32) -> HRESULT {
    xasync::XAsyncGetStatus(block, false)
}
extern "system" fn array_status(
    _: This,
    block: *mut AsyncBlock,
    _: u32,
    _: *mut c_void,
) -> HRESULT {
    xasync::XAsyncGetStatus(block, false)
}

extern "system" fn purchase_ui_async(
    _: This,
    _: Handle,
    _store_id: *const c_char,
    _name: *const c_char,
    _json: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "purchase page")
}
extern "system" fn product_page_ui_async(
    _: This,
    _: Handle,
    _store_id: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "product page")
}
extern "system" fn redeem_ui_async(
    _: This,
    _: Handle,
    _token: *const c_char,
    _allowed: *const *const c_char,
    _allowed_count: usize,
    _disallow_csv: bool,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "redeem page")
}
extern "system" fn updates_async(_: This, _: Handle, block: *mut AsyncBlock) -> HRESULT {
    offline(block, "update check")
}
/// Installing updates or packages by id alike.
extern "system" fn install_async(
    _: This,
    _: Handle,
    _ids: *const *const c_char,
    _count: usize,
    block: *mut AsyncBlock,
) -> HRESULT {
    offline(block, "package download")
}

/// A store id is the installation's own when it is the running title's,
/// and the manifest has one (an empty id matches nothing).
fn installed_package(title: &runtime::Title, store_id: Option<&str>) -> Option<String> {
    let id = store_id?;
    (!title.store_id.is_empty() && id == title.store_id).then(xpackage::identifier)
}

/// The game's own store id maps to its installation; nothing else is
/// installed, and any other id gets the documented E_INVALIDARG (not
/// XPackage's "could not resolve" error).
extern "system" fn package_identifier(
    _: This,
    store_id: *const c_char,
    size: usize,
    buffer: *mut u8,
) -> HRESULT {
    let Some(identifier) = installed_package(&runtime::title(), text(store_id).as_deref()) else {
        return E_INVALIDARG;
    };
    put_string(&identifier, size, buffer, std::ptr::null_mut())
}

/// Licences never change offline, so these callbacks never run.
extern "system" fn register(
    _: This,
    _: Handle,
    _queue: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    put(out, token())
}
extern "system" fn unregister(_: This, _: Handle, _token: u64, _wait: bool) -> bool {
    true
}

static VTABLE: Methods<77> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    create_context as *const (),             // 0x18
    close as *const (),                      // 0x20
    associated_products_async as *const (),  // 0x28
    query_result as *const (),               // 0x30
    missing::<7> as *const (),               // 0x38 products by id
    missing::<8> as *const (),               // 0x40
    entitled_products_async as *const (),    // 0x48
    query_result as *const (),               // 0x50
    missing::<11> as *const (),              // 0x58 product for the game
    missing::<12> as *const (),              // 0x60
    missing::<13> as *const (),              // 0x68 product for a package
    missing::<14> as *const (),              // 0x70
    enumerate_products as *const (),         // 0x78
    has_more_pages as *const (),             // 0x80
    next_page_async as *const (),            // 0x88
    query_result as *const (),               // 0x90
    close as *const (),                      // 0x98 a product query
    license_for_package_async as *const (),  // 0xa0
    license_result as *const (),             // 0xa8
    is_license_valid as *const (),           // 0xb0
    close as *const (),                      // 0xb8 a licence
    missing::<24> as *const (),              // 0xc0 can acquire for store id
    missing::<25> as *const (),              // 0xc8
    missing::<26> as *const (),              // 0xd0 can acquire for package
    missing::<27> as *const (),              // 0xd8
    game_license_async as *const (),         // 0xe0
    game_license_result as *const (),        // 0xe8
    missing::<30> as *const (),              // 0xf0 add-on licences
    missing::<31> as *const (),              // 0xf8
    missing::<32> as *const (),              // 0x100
    consumable_balance_async as *const (),   // 0x108
    count_status as *const (),               // 0x110
    report_fulfillment_async as *const (),   // 0x118
    count_status as *const (),               // 0x120
    user_id_async as *const (),              // 0x128 collections id
    size_status as *const (),                // 0x130
    string_status as *const (),              // 0x138
    user_id_async as *const (),              // 0x140 purchase id
    size_status as *const (),                // 0x148
    string_status as *const (),              // 0x150
    missing::<43> as *const (),              // 0x158 licence token
    missing::<44> as *const (),              // 0x160
    missing::<45> as *const (),              // 0x168
    missing::<46> as *const (),              // 0x170
    missing::<47> as *const (),              // 0x178
    missing::<48> as *const (),              // 0x180
    purchase_ui_async as *const (),          // 0x188
    status as *const (),                     // 0x190
    missing::<51> as *const (),              // 0x198 rate and review
    missing::<52> as *const (),              // 0x1a0
    redeem_ui_async as *const (),            // 0x1a8
    status as *const (),                     // 0x1b0
    updates_async as *const (),              // 0x1b8
    count_status as *const (),               // 0x1c0
    array_status as *const (),               // 0x1c8
    missing::<58> as *const (),              // 0x1d0 download updates
    missing::<59> as *const (),              // 0x1d8
    install_async as *const (),              // 0x1e0 updates
    status as *const (),                     // 0x1e8
    install_async as *const (),              // 0x1f0 packages by store id
    count_status as *const (),               // 0x1f8
    array_status as *const (),               // 0x200
    package_identifier as *const (),         // 0x208
    register as *const (),                   // 0x210 game licence changed
    unregister as *const (),                 // 0x218
    register as *const (),                   // 0x220 package licence lost
    unregister as *const (),                 // 0x228
    missing::<70> as *const (),              // 0x230
    license_for_durables_async as *const (), // 0x238
    license_result as *const (),             // 0x240
    missing::<73> as *const (),              // 0x248 associated products page
    missing::<74> as *const (),              // 0x250
    product_page_ui_async as *const (),      // 0x258
    status as *const (),                     // 0x260
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn game_license_result_bytes_follow_the_abi_and_zero_padding() {
        let license = GameLicense {
            sku_store_id: [0x5a; 18],
            is_active: true,
            is_trial_owned_by_this_user: false,
            is_disc_license: true,
            is_trial: true,
            trial_seconds_remaining: 0x1234_5678,
            trial_unique_id: [0x6b; 64],
            expiration: -0x0102_0304_0506_0708,
        };
        let payload = xasync::bytes(license);
        assert_eq!(payload.len(), 104);
        assert_eq!(&payload[..18], &[0x5a; 18]);
        assert_eq!(&payload[18..24], &[1, 0, 1, 1, 0, 0]);
        assert_eq!(&payload[24..28], &0x1234_5678u32.to_ne_bytes());
        assert_eq!(&payload[28..92], &[0x6b; 64]);
        assert_eq!(&payload[92..96], &[0; 4]);
        assert_eq!(&payload[96..], &license.expiration.to_ne_bytes());

        let mut b = block();
        assert_eq!(
            xasync::finished_with(&mut b, (&raw const GAME_LICENSE).cast(), Ok(payload)),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        let mut result = std::mem::MaybeUninit::<GameLicense>::uninit();
        assert_eq!(
            game_license_result(&OBJECT, &mut b, result.as_mut_ptr()),
            S_OK
        );
        // SAFETY: the successful result initialized every field.
        let result = unsafe { result.assume_init() };
        assert_eq!(result.sku_store_id, license.sku_store_id);
        assert_eq!(result.is_active, license.is_active);
        assert_eq!(
            result.is_trial_owned_by_this_user,
            license.is_trial_owned_by_this_user
        );
        assert_eq!(result.is_disc_license, license.is_disc_license);
        assert_eq!(result.is_trial, license.is_trial);
        assert_eq!(
            result.trial_seconds_remaining,
            license.trial_seconds_remaining
        );
        assert_eq!(result.trial_unique_id, license.trial_unique_id);
        assert_eq!(result.expiration, license.expiration);
    }

    fn block() -> AsyncBlock {
        AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        }
    }

    /// Runs `begin` on `b` and checks its call finished at once, as the title
    /// then reads it: the status word, and a result size (none if it failed).
    /// The block stays where it is, since a call is keyed by its address.
    fn ran(b: &mut AsyncBlock, begin: impl FnOnce(*mut AsyncBlock) -> HRESULT, expected: HRESULT) {
        assert_eq!(begin(b), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(b, true), expected);
        assert_eq!(b.internal[1], expected as u32 as usize);
        if expected < 0 {
            assert_eq!(b.internal[2], 0, "a failed call has no result");
        }
    }

    #[test]
    fn a_package_licence_names_the_installed_package() {
        type Ask = extern "system" fn(This, Handle, *const c_char, *mut AsyncBlock) -> HRESULT;
        type Result = extern "system" fn(This, *mut AsyncBlock, *mut Handle) -> HRESULT;
        type Valid = extern "system" fn(This, Handle) -> bool;
        let this: This = &OBJECT;
        let context = handle(&CONTEXT);
        let ask = method::<Ask>(&OBJECT, 0xa0);
        let result = method::<Result>(&OBJECT, 0xa8);
        let valid = method::<Valid>(&OBJECT, 0xb0);
        let licence_for = |package: *const c_char| {
            let mut b = block();
            ran(&mut b, |b| ask(this, context, package, b), S_OK);
            let mut license: Handle = std::ptr::null();
            assert_eq!(result(this, &mut b, &mut license), S_OK);
            assert!(!license.is_null());
            valid(this, license)
        };

        let ours = format!("{}\0", xpackage::identifier());
        assert!(licence_for(ours.as_ptr().cast()));
        assert!(!licence_for(c"another-package".as_ptr()));
        assert!(!licence_for(std::ptr::null()));

        // Only a licence we handed out is valid.
        assert!(!valid(this, std::ptr::null()));
        assert!(!valid(this, handle(&CONTEXT)));
        assert!(!valid(this, handle(&QUERY)));
        assert!(valid(this, handle(&VALID)));
    }

    #[test]
    fn the_game_license_is_full_and_names_the_first_sku() {
        type Begin = extern "system" fn(This, Handle, *mut AsyncBlock) -> HRESULT;
        type Result = extern "system" fn(This, *mut AsyncBlock, *mut GameLicense) -> HRESULT;
        let this: This = &OBJECT;
        let mut b = block();
        ran(
            &mut b,
            |b| method::<Begin>(&OBJECT, 0xe0)(this, handle(&CONTEXT), b),
            S_OK,
        );
        let result = method::<Result>(&OBJECT, 0xe8);

        let mut size = 0;
        assert_eq!(xasync::XAsyncGetResultSize(&mut b, &mut size), S_OK);
        assert_eq!(size, size_of::<GameLicense>());
        // No room, no result; the call goes on.
        assert_eq!(
            result(this, &mut b, std::ptr::null_mut()),
            xasync::E_NOT_SUFFICIENT_BUFFER
        );

        let mut license = std::mem::MaybeUninit::<GameLicense>::zeroed();
        assert_eq!(result(this, &mut b, license.as_mut_ptr()), S_OK);
        // SAFETY: written by the call above.
        let license = unsafe { license.assume_init() };
        assert!(license.is_active);
        assert!(
            !license.is_trial && !license.is_disc_license && !license.is_trial_owned_by_this_user
        );
        assert_eq!(
            (license.trial_seconds_remaining, license.expiration),
            (0, 0)
        );
        assert_eq!(license.trial_unique_id, [0; 64]);
        let end = license.sku_store_id.iter().position(|b| *b == 0).unwrap();
        let sku = std::str::from_utf8(&license.sku_store_id[..end]).unwrap();
        assert!(sku.ends_with("/0010"), "{sku:?}");

        // The result ends the call, as the query's did.
        let mut after = std::mem::MaybeUninit::<GameLicense>::zeroed();
        assert_eq!(
            result(this, &mut b, after.as_mut_ptr()),
            E_ILLEGAL_METHOD_CALL
        );
    }

    #[test]
    fn the_store_id_maps_to_the_installation() {
        type Ident = extern "system" fn(This, *const c_char, usize, *mut u8) -> HRESULT;
        let this: This = &OBJECT;
        let ident = method::<Ident>(&OBJECT, 0x208);
        let mut buffer = [0u8; 33];
        // This test's title has no store id, so nothing resolves.
        assert_eq!(
            ident(this, std::ptr::null(), buffer.len(), buffer.as_mut_ptr()),
            E_INVALIDARG
        );
        assert_eq!(
            ident(this, c"9XYZ".as_ptr(), buffer.len(), buffer.as_mut_ptr()),
            E_INVALIDARG
        );
        assert_eq!(buffer, [0u8; 33], "nothing written");

        // The mapping itself, for a title whose manifest has one.
        let title = runtime::Title {
            id: 0,
            store_id: "9XYZ".to_owned(),
            ..Default::default()
        };
        let package = installed_package(&title, Some("9XYZ")).unwrap();
        assert!(
            !package.is_empty() && package.chars().all(|c| c.is_ascii_hexdigit()),
            "{package:?}"
        );
        assert_eq!(installed_package(&title, Some("other")), None);
        assert_eq!(installed_package(&title, Some("")), None);
        assert_eq!(installed_package(&title, None), None);
        let unnamed = runtime::Title::default();
        assert_eq!(
            installed_package(&unnamed, Some("9XYZ")),
            None,
            "a manifest without a store id names nothing"
        );
    }

    #[test]
    fn offline_async_calls_report_network_error_from_their_result_slots() {
        assert!(VTABLE.0.len() > 0x260 / 8);

        let this: This = &OBJECT;
        type StoreIdAsync =
            extern "system" fn(This, Handle, *const c_char, *mut AsyncBlock) -> HRESULT;
        type ReportFulfillmentAsync = extern "system" fn(
            This,
            Handle,
            *const c_char,
            u32,
            *const c_void,
            *mut AsyncBlock,
        ) -> HRESULT;
        type PurchaseUiAsync = extern "system" fn(
            This,
            Handle,
            *const c_char,
            *const c_char,
            *const c_char,
            *mut AsyncBlock,
        ) -> HRESULT;
        type RedeemUiAsync = extern "system" fn(
            This,
            Handle,
            *const c_char,
            *const *const c_char,
            usize,
            bool,
            *mut AsyncBlock,
        ) -> HRESULT;
        type StatusResult = extern "system" fn(This, *mut AsyncBlock) -> HRESULT;
        type CountResult = extern "system" fn(This, *mut AsyncBlock, *mut u32) -> HRESULT;

        let mut b = block();
        assert_eq!(
            method::<StoreIdAsync>(&OBJECT, 0x258)(
                this,
                handle(&CONTEXT),
                c"9DUMMY".as_ptr(),
                &mut b,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        assert_eq!(
            method::<StatusResult>(&OBJECT, 0x260)(this, &mut b),
            NETWORK_ERROR
        );

        let mut b = block();
        let mut remaining = 7u32;
        assert_eq!(
            method::<ReportFulfillmentAsync>(&OBJECT, 0x118)(
                this,
                handle(&CONTEXT),
                c"9DUMMY".as_ptr(),
                1,
                std::ptr::null(),
                &mut b,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        assert_eq!(
            method::<CountResult>(&OBJECT, 0x120)(this, &mut b, &mut remaining),
            NETWORK_ERROR
        );
        assert_eq!(remaining, 7);

        let mut b = block();
        assert_eq!(
            method::<PurchaseUiAsync>(&OBJECT, 0x188)(
                this,
                handle(&CONTEXT),
                c"9DUMMY".as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                &mut b,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        assert_eq!(
            method::<StatusResult>(&OBJECT, 0x190)(this, &mut b),
            NETWORK_ERROR
        );

        let mut b = block();
        assert_eq!(
            method::<RedeemUiAsync>(&OBJECT, 0x1a8)(
                this,
                handle(&CONTEXT),
                c"DUMMY-TOKEN".as_ptr(),
                std::ptr::null(),
                0,
                false,
                &mut b,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        assert_eq!(
            method::<StatusResult>(&OBJECT, 0x1b0)(this, &mut b),
            NETWORK_ERROR
        );

        let mut b = block();
        let mut balance = 3u32;
        assert_eq!(
            method::<StoreIdAsync>(&OBJECT, 0x108)(
                this,
                handle(&CONTEXT),
                c"9DUMMY".as_ptr(),
                &mut b,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        assert_eq!(
            method::<CountResult>(&OBJECT, 0x110)(this, &mut b, &mut balance),
            NETWORK_ERROR
        );
        assert_eq!(balance, 3);
    }

    #[test]
    fn the_game_is_licensed_and_the_store_is_offline() {
        let this: This = &OBJECT;
        assert_eq!(size_of::<GameLicense>(), 104);
        type Begin = extern "system" fn(This, Handle, *mut AsyncBlock) -> HRESULT;
        type Result = extern "system" fn(This, *mut AsyncBlock, *mut GameLicense) -> HRESULT;
        let mut b = block();
        assert_eq!(
            method::<Begin>(&OBJECT, 0xe0)(this, handle(&CONTEXT), &mut b),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        let mut license = std::mem::MaybeUninit::<GameLicense>::zeroed();
        assert_eq!(
            method::<Result>(&OBJECT, 0xe8)(this, &mut b, license.as_mut_ptr()),
            S_OK
        );
        // SAFETY: written by the call.
        let license = unsafe { license.assume_init() };
        assert!(license.is_active && !license.is_trial);

        let mut b = block();
        assert_eq!(
            method::<Begin>(&OBJECT, 0x1b8)(this, handle(&CONTEXT), &mut b),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), NETWORK_ERROR);
        type Count = extern "system" fn(This, *mut AsyncBlock, *mut u32) -> HRESULT;
        assert_eq!(
            method::<Count>(&OBJECT, 0x1c0)(this, &mut b, &mut 0),
            NETWORK_ERROR
        );

        type Durable = extern "system" fn(This, Handle, *const c_char, *mut AsyncBlock) -> HRESULT;
        type License = extern "system" fn(This, *mut AsyncBlock, *mut Handle) -> HRESULT;
        let mut b = block();
        let durable = method::<Durable>(&OBJECT, 0x238);
        assert_eq!(
            durable(this, handle(&CONTEXT), c"9XYZ".as_ptr(), &mut b),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        let mut l = std::ptr::null();
        assert_eq!(
            method::<License>(&OBJECT, 0x240)(this, &mut b, &mut l),
            S_OK
        );
        assert!(!method::<extern "system" fn(This, Handle) -> bool>(
            &OBJECT, 0xb0
        )(this, l));
    }

    #[test]
    fn a_store_id_without_a_package_is_an_invalid_argument() {
        type Query = extern "system" fn(This, *const c_char, usize, *mut u8) -> HRESULT;
        let query = method::<Query>(&OBJECT, 0x208);
        let mut id = [0u8; 33];
        assert_eq!(
            query(&OBJECT, c"9NOTINSTALLED".as_ptr(), 33, id.as_mut_ptr()),
            E_INVALIDARG
        );
    }
}
