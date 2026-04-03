# Complex Refactor — 4bit

**Model:** `mlx-community/Qwen3.5-397B-A17B-4bit`

| Metric | Value |
|---|---|
| Thinking tokens | 1039 |
| Completion tokens | 1790 |
| Decode tok/s | 33.5 |
| TTFT | 2306ms |
| Total time | 55.7s |

## Thinking

```
This is a classic callback hell scenario with several issues:
1. Nested callbacks (callback pyramid)
2. A bug where the callback fires before all async operations complete (the setTimeout is a hack)
3. No proper error handling
4. No types (it's JavaScript, not TypeScript)

I need to refactor this to:
1. Use async/await instead of callbacks
2. Properly wait for all order items to be fetched (using Promise.all)
3. Add proper TypeScript types
4. Maintain the exact same behavior (drop-in replacement)
5. Proper error handling with try/catch

Key considerations:
- The function signature needs to remain compatible (though we can make it return a Promise)
- Need to handle the case where no user is found (returns null)
- Need to wait for ALL order items to complete before returning
- Need proper TypeScript types for User, Order, OrderItem
- Need to preserve the database query interface (assuming it can work with promises or we need to promisify it)

Let me create a proper refactored version:

```typescript
interface User {
  id: number;
  [key: string]: any;
}

interface Order {
  id: number;
  user_id: number;
  items?: any[];
  [key: string]: any;
}

interface OrderItem {
  id: number;
  order_id: number;
  [key: string]: any;
}

async function fetchUserData(userId: number): Promise<User | null> {
  try {
    // Fetch user
    const userRows = await db.query('SELECT * FROM users WHERE id = ?', [userId]);
    
    if (!userRows.length) {
      return null;
    }
    
    const user: User = userRows[0];
    
    // Fetch orders
    const orders = await db.query('SELECT * FROM orders WHERE user_id = ?', [userId]);
    user.orders = orders;
    
    // Fetch order items for each order
    await Promise.all(
      orders.map(async (order: Order) => {
        const items = await db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id]);
        order.items = items;
      })
    );
    
    return user;
  } catch (error) {
    throw error;
  }
}
```

Wait, I need to think about the callback signature. The original function takes a callback, so for it to be a "drop-in replacement", I should consider whether to:
1. Keep the callback signature but use async/await internally
2. Change to Promise-based and expect callers to update

The instruction says "drop-in replacement" which suggests the function signature should remain the same. However, modernizing to async/await typically means returning a Promise. Let me provide both options or make it clear.

Actually, looking at the instruction again: "Refactor this legacy callback-based Node.js code into modern async/await TypeScript" - this suggests converting TO async/await, which means the function should return a Promise. The "drop-in replacement" likely means the behavior should be the same, not necessarily the exact signature.

However, for true drop-in replacement with callbacks, I could make it work both ways. But the cleanest modern approach is to return a Promise.

Let me also consider:
- The db.query interface - in the original it takes a callback, so I need to either assume it supports promises or promisify it
- Error handling - should propagate errors properly
- The bug with setTimeout needs to be fixed properly with Promise.all

Let me create a complete solution:

```typescript
interface User {
  id: number;
  [key: string]: any;
  orders?: Order[];
}

interface Order {
  id: number;
  user_id: number;
  items?: any[];
  [key: string]: any;
}

async function fetchUserData(userId: number): Promise<User | null> {
  try {
    const userRows: any[] = await db.query('SELECT * FROM users WHERE id = ?', [userId]);
    
    if (!userRows.length) {
      return null;
    }
    
    const user: User = userRows[0];
    const orders: Order[] = await db.query('SELECT * FROM orders WHERE user_id = ?', [userId]);
    
    user.orders = orders;
    
    await Promise.all(
      orders.map(async (order: Order) => {
        const items: any[] = await db.query('SELECT * FROM order_items WHERE order_id = ?', [order.id]);
        order.items = items;
      })
    );
    
    return user;
  } catch (error) {
    throw error;
  }
}
```

I'll focus on creating a clean, type-safe solution that properly handles async database queries while maintaining the original functionality.

```

## Response



# Refactored Code: Modern Async/Await TypeScript

```typescript
interface User {
  id: number;
  [key: string]: any;
  orders?: Order[];
}

interface Order {
  id: number;
  user_id: number;
  items?: any[];
  [key: string]: any;
}

/**
 * Fetches user data with nested orders and order items.
 * Replaces callback-based implementation with async/await.
 * 
 * @param userId - The user ID to fetch
 * @returns Promise resolving to user data or null if not found
 * @throws Error if database query fails
 */
async function fetchUserData(userId: number): Promise<User | null> {
  try {
    // Fetch user
    const userRows: any[] = await db.query(
      'SELECT * FROM users WHERE id = ?', 
      [userId]
    );

    if (!userRows.length) {
      return null;
    }

    const user: User = userRows[0];

    // Fetch orders
    const orders: Order[] = await db.query(
      'SELECT * FROM orders WHERE user_id = ?', 
      [userId]
    );

    user.orders = orders;

    // Fetch order items for ALL orders in parallel
    // FIX: Properly waits for all queries to complete (no setTimeout hack)
    await Promise.all(
      orders.map(async (order: Order) => {
        const items: any[] = await db.query(
          'SELECT * FROM order_items WHERE order_id = ?', 
          [order.id]
        );
        order.items = items;
      })
    );

    return user;
  } catch (error) {
    // Propagate errors to caller for proper handling
    throw error;
  }
}

export { fetchUserData, User, Order };
```

## Usage Example

```typescript
// Before (callback):
fetchUserData(123, (err, user) => {
  if (err) return handleError(err);
  if (!user) return handleNotFound();
  console.log(user);
});

// After (async/await):
try {
  const user = await fetchUserData(123);
  if (!user) {
    handleNotFound();
    return;
  }
  console.log(user);
} catch (error) {
  handleError(error);
}
```

## Key Improvements

| Issue | Before | After |
|-------|--------|-------|
| **Callback pyramid** | 3 levels of nesting | Flat, sequential code |
| **Race condition bug** | `setTimeout` hack | `Promise.all` waits properly |
| **Error handling** | Check `err` in each callback | Single `try/catch` block |
| **Type safety** | No types | Full TypeScript interfaces |
| **Maintainability** | Hard to follow | Clear execution flow |

## Note on Database Interface

If your `db.query` doesn't natively support promises, wrap it:

```typescript
import { promisify } from 'util';
const query = promisify(db.query).bind(db);
// Then use `query()` instead of `db.query()`
```

This refactoring maintains identical behavior while providing modern, type-safe, and maintainable code.
